"""
Gemini API wrapper for lemlem.

Handles Gemini's native API format and automatic thought signature management.
Converts between OpenAI format (used internally by lemlem) and Gemini's native format.
"""

import logging
import concurrent.futures
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai
    from google.genai import types

logger = logging.getLogger(__name__)


class GeminiWrapper:
    """Wrapper for Gemini API using the native google-genai library."""

    def __init__(self, api_key: str, model_name: str):
        """
        Initialize Gemini wrapper.

        Args:
            api_key: Gemini API key
            model_name: Model name (e.g., "gemini-3-pro-preview")
                       Should NOT include "models/" prefix when using native SDK
        """
        # Lazy import to avoid dependency issues
        try:
            from google import genai as _genai
            from google.genai import types as _types
            self._genai = _genai
            self._types = _types
        except ImportError as e:
            raise ImportError(
                "google-genai library is required for Gemini support. "
                "Install it with: pip install google-genai"
            ) from e

        self.client = self._genai.Client(api_key=api_key)
        # Remove 'models/' prefix if present
        self.model_name = model_name.replace("models/", "")
        # Store thought signatures and function names by tool_call_id for multi-turn conversations
        self._thought_signatures: Dict[str, bytes] = {}
        self._function_names: Dict[str, str] = {}
        self._tool_call_counter: int = 0

    def convert_openai_messages_to_gemini(
        self, messages: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Convert OpenAI message format to Gemini native format.

        Args:
            messages: OpenAI-style messages

        Returns:
            List of Gemini Content objects
        """
        Content = self._types.Content
        Part = self._types.Part
        FunctionCall = self._types.FunctionCall
        FunctionResponse = self._types.FunctionResponse

        contents = []

        for msg in messages:
            role = msg.get("role")
            content_data = msg.get("content")

            # Handle different message types
            if role == "system":
                # Gemini doesn't have a system role, prepend as user message
                contents.append(
                    Content(
                        role="user",
                        parts=[Part(text=f"System: {content_data}")]
                    )
                )

            elif role == "user":
                parts = []
                if isinstance(content_data, str):
                    parts.append(Part(text=content_data))
                elif isinstance(content_data, list):
                    for part in content_data:
                        if part.get("type") == "text":
                            parts.append(Part(text=part.get("text", "")))
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {})
                            url = image_url.get("url", "")
                            if url.startswith("data:image/"):
                                # Handle base64 data URL
                                try:
                                    import base64
                                    header, encoded = url.split(",", 1)
                                    mime_type = header.split(";")[0].split(":")[1]
                                    data = base64.b64decode(encoded)
                                    parts.append(Part.from_bytes(data=data, mime_type=mime_type))
                                except Exception as e:
                                    logger.warning(f"Failed to parse base64 image_url: {e}")
                            else:
                                # For remote URLs, Gemini native SDK doesn't support them directly in parts
                                # (must be uploaded or passed as bytes)
                                # We'll skip for now or we could download it here
                                logger.warning(f"Remote image URLs not supported in Gemini native wrapper yet: {url}")
                        elif part.get("type") == "image_path":
                            # Custom type for local files
                            path_str = part.get("path", "")
                            if path_str:
                                from pathlib import Path
                                path = Path(path_str)
                                if path.exists():
                                    mime_type = "image/png"  # Default
                                    suffix = path.suffix.lower()
                                    if suffix in {".jpg", ".jpeg"}: mime_type = "image/jpeg"
                                    elif suffix == ".webp": mime_type = "image/webp"
                                    elif suffix == ".gif": mime_type = "image/gif"
                                    
                                    data = path.read_bytes()
                                    parts.append(Part.from_bytes(data=data, mime_type=mime_type))
                contents.append(Content(role="user", parts=parts))

            elif role == "assistant":
                # Handle assistant messages with potential tool calls
                tool_calls = msg.get("tool_calls")
                parts = []

                if tool_calls:
                    # Convert tool calls to Gemini function_call format
                    for tool_call in tool_calls:
                        # Handle both dict and object formats
                        if isinstance(tool_call, dict):
                            func = tool_call.get("function", {})
                            tool_call_id = tool_call.get("id")
                            # Extract persisted thought_signature (base64 encoded)
                            persisted_thought_sig = tool_call.get("thought_signature")
                        else:
                            func = getattr(tool_call, "function", None)
                            tool_call_id = getattr(tool_call, "id", None)
                            persisted_thought_sig = getattr(tool_call, "thought_signature", None)

                        import json
                        if isinstance(func, dict):
                            func_name = func.get("name")
                            args_str = func.get("arguments", "{}")
                        else:
                            func_name = getattr(func, "name", None) if func else None
                            args_str = getattr(func, "arguments", "{}") if func else "{}"

                        # SKIP tool calls with empty names to avoid Gemini 400 errors
                        if not func_name:
                            logger.warning(f"Skipping assistant tool call with empty name in history: {tool_call}")
                            continue

                        # Restore function_name mapping for tool responses
                        if tool_call_id and func_name:
                            self._function_names[tool_call_id] = func_name

                        args = json.loads(args_str) if isinstance(args_str, str) else args_str

                        # Get thought_signature: first check persisted (from JSON), then in-memory cache
                        thought_sig = None
                        if persisted_thought_sig:
                            # Decode from base64 (persisted format)
                            import base64
                            try:
                                thought_sig = base64.b64decode(persisted_thought_sig)
                                # Also cache it for potential future use within this session
                                if tool_call_id:
                                    self._thought_signatures[tool_call_id] = thought_sig
                                logger.debug(f"Restored thought_signature from persisted data for {tool_call_id}")
                            except Exception as e:
                                logger.warning(f"Failed to decode persisted thought_signature: {e}")

                        # Fall back to in-memory cache if not found in persisted data
                        if not thought_sig and tool_call_id:
                            thought_sig = self._thought_signatures.get(tool_call_id)

                        # Create the Part with function_call and thought_signature if available
                        part_kwargs = {
                            "function_call": FunctionCall(
                                name=func_name,
                                args=args
                            )
                        }
                        if thought_sig:
                            part_kwargs["thought_signature"] = thought_sig

                        parts.append(Part(**part_kwargs))

                if not parts and content_data:
                    parts.append(Part(text=content_data))
                elif parts and content_data:
                    # Support both text and tool calls in the same message
                    parts.insert(0, Part(text=content_data))

                if parts:
                    contents.append(Content(role="model", parts=parts))

            elif role == "tool":
                # Convert tool response to Gemini function_response format
                import json
                tool_call_id = msg.get("tool_call_id")
                result = json.loads(content_data) if isinstance(content_data, str) else content_data

                # Get function name from stored mapping
                func_name = self._function_names.get(tool_call_id, "unknown")
                if func_name == "unknown":
                    logger.warning(f"Could not find function name for tool_call_id {tool_call_id}")

                contents.append(
                    Content(
                        role="user",
                        parts=[
                            Part(
                                function_response=FunctionResponse(
                                    name=func_name,
                                    response=result
                                )
                            )
                        ]
                    )
                )

        return contents

    def _choose_union_type(self, types: List[Any]) -> Optional[str]:
        normalized = [str(t).lower() for t in types if isinstance(t, str) and t]
        if not normalized:
            return None
        non_null = [t for t in normalized if t != "null"]
        if not non_null:
            return "null"
        if len(non_null) == 1:
            return non_null[0]
        # Gemini function schema expects a single type. Prefer string when mixed.
        if "string" in non_null:
            return "string"
        if "object" in non_null:
            return "object"
        if "array" in non_null:
            return "array"
        if "number" in non_null:
            return "number"
        if "integer" in non_null:
            return "integer"
        if "boolean" in non_null:
            return "boolean"
        return non_null[0]

    def _select_union_option(self, options: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(options, list):
            return None
        dict_options = [opt for opt in options if isinstance(opt, dict)]
        if not dict_options:
            return None
        # Prefer non-null branches and simpler scalar branches for compatibility.
        for candidate in dict_options:
            ctype = candidate.get("type")
            if isinstance(ctype, str) and ctype.lower() != "null":
                return candidate
        return dict_options[0]

    def _sanitize_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove fields from JSON Schema that Gemini doesn't support.

        Args:
            schema: JSON Schema dict

        Returns:
            Sanitized schema dict
        """
        if not isinstance(schema, dict):
            return schema

        # Fields that Gemini doesn't support
        unsupported_fields = {
            'additionalProperties',
            '$schema',
            '$ref',
            '$id',
            'definitions',
        }

        working = dict(schema)

        # Gemini tool schemas do not support oneOf/anyOf directly.
        for union_key in ("oneOf", "anyOf"):
            selected = self._select_union_option(working.get(union_key))
            if selected:
                for merge_key, merge_value in selected.items():
                    if merge_key not in working:
                        working[merge_key] = merge_value
                    elif merge_key in {"description"} and isinstance(working[merge_key], str):
                        if isinstance(merge_value, str) and merge_value not in working[merge_key]:
                            working[merge_key] = f"{working[merge_key]} {merge_value}".strip()
            working.pop(union_key, None)

        # Gemini expects `type` to be a single enum string, not a list.
        raw_type = working.get("type")
        if isinstance(raw_type, list):
            chosen = self._choose_union_type(raw_type)
            if chosen:
                working["type"] = chosen
            else:
                working.pop("type", None)

        normalized_type = working.get("type")
        if isinstance(normalized_type, str):
            normalized_type = normalized_type.lower()

            # Keep schema keywords consistent with the resolved type.
            if normalized_type != "array":
                working.pop("items", None)
                working.pop("minItems", None)
                working.pop("maxItems", None)

            if normalized_type != "object":
                working.pop("properties", None)
                working.pop("required", None)

        # Create a copy and remove unsupported fields
        sanitized = {}
        for key, value in working.items():
            if key in unsupported_fields:
                continue

            # Recursively sanitize nested objects
            if isinstance(value, dict):
                sanitized[key] = self._sanitize_json_schema(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_json_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    def convert_openai_tools_to_gemini(
        self, tools: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Convert OpenAI tool format to Gemini Tool format.

        Args:
            tools: OpenAI-style tools

        Returns:
            List of Gemini Tool objects
        """
        Tool = self._types.Tool
        FunctionDeclaration = self._types.FunctionDeclaration

        function_declarations = []

        for idx, tool in enumerate(tools):
            if tool.get("type") == "function":
                # Handle both FLAT and NESTED formats
                # FLAT: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
                # NESTED: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
                if "function" in tool:
                    # Nested format
                    func = tool.get("function", {})
                    func_name = func.get("name")
                    func_desc = func.get("description", "")
                    func_params = func.get("parameters", {})
                else:
                    # Flat format
                    func_name = tool.get("name")
                    func_desc = tool.get("description", "")
                    func_params = tool.get("parameters", {})

                # Validate function name
                if not func_name:
                    logger.error(f"Tool {idx} has empty/None name: {tool}")
                    raise ValueError(f"Tool {idx} has empty or None name")

                if not isinstance(func_name, str):
                    logger.error(f"Tool {idx} name is not a string: {type(func_name)} = {func_name}")
                    func_name = str(func_name)

                # Sanitize parameters to remove Gemini-unsupported fields
                sanitized_params = self._sanitize_json_schema(func_params)

                function_declarations.append(
                    FunctionDeclaration(
                        name=func_name,
                        description=func_desc,
                        parameters=sanitized_params
                    )
                )

        return [Tool(function_declarations=function_declarations)] if function_declarations else []

    def generate_content(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini API with OpenAI-compatible interface.

        Args:
            messages: OpenAI-style messages
            tools: OpenAI-style tools
            temperature: Temperature for generation
            max_tokens: Max output tokens
            **kwargs: Additional Gemini-specific params

        Returns:
            OpenAI-style response dict
        """

        # Convert to Gemini format
        contents = self.convert_openai_messages_to_gemini(messages)
        gemini_tools = self.convert_openai_tools_to_gemini(tools) if tools else None

        # Build config
        config_params = {}
        if temperature is not None:
            config_params["temperature"] = temperature
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens

        # Add tools
        if gemini_tools:
            config_params["tools"] = gemini_tools

        # Call Gemini API
        try:
            GenerateContentConfig = self._types.GenerateContentConfig
            # Extract timeout from kwargs if present, as it's passed to the client method, not the config
            request_timeout = kwargs.pop("timeout", None)
            
            call_kwargs = {
                "model": self.model_name,
                "contents": contents,
                "config": GenerateContentConfig(**config_params) if config_params else None
            }
            
            if request_timeout is not None:
                # google-genai expects timeout in milliseconds via types.HttpOptions
                # Convert from seconds to milliseconds, with minimum of 10s (10000ms) per API requirements
                timeout_ms = max(10000, int(request_timeout * 1000))
                HttpOptions = self._types.HttpOptions
                call_kwargs["config"] = call_kwargs["config"] or GenerateContentConfig()
                call_kwargs["config"].http_options = HttpOptions(timeout=timeout_ms)

            hard_timeout = None
            if request_timeout is not None:
                try:
                    hard_timeout = max(10.0, float(request_timeout) + 15.0)
                except (TypeError, ValueError):
                    hard_timeout = None

            if hard_timeout is None:
                response = self.client.models.generate_content(**call_kwargs)
            else:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(self.client.models.generate_content, **call_kwargs)
                try:
                    response = future.result(timeout=hard_timeout)
                except concurrent.futures.TimeoutError as exc:
                    future.cancel()
                    raise TimeoutError(
                        f"Gemini generate_content timed out after {hard_timeout:.1f}s"
                    ) from exc
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

            # Convert response to OpenAI format
            return self._convert_gemini_response_to_openai(response)

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

    def _convert_gemini_response_to_openai(self, response: Any) -> Dict[str, Any]:
        """
        Convert Gemini response to OpenAI format.

        Args:
            response: Gemini GenerateContentResponse

        Returns:
            OpenAI-style response dict
        """
        if not response.candidates:
            return {
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

        candidate = response.candidates[0]
        content = candidate.content

        # Extract text and tool calls
        message = {"role": "assistant"}
        text_parts = []
        tool_calls = []

        # Safety check: ensure parts exists
        if not content.parts:
            logger.warning(f"Gemini response has no parts. Candidate finish_reason: {getattr(candidate, 'finish_reason', 'unknown')}, content: {content}")
            message["content"] = ""
            usage_meta = response.usage_metadata
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": getattr(usage_meta, "prompt_token_count", None) or 0 if usage_meta else 0,
                    "completion_tokens": getattr(usage_meta, "candidates_token_count", None) or 0 if usage_meta else 0,
                    "total_tokens": getattr(usage_meta, "total_token_count", None) or 0 if usage_meta else 0
                },
                "model": self.model_name
            }

        for part in content.parts:
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                import json
                
                # Check for empty function name which causes errors in subsequent turns
                if not part.function_call.name:
                    logger.warning("Gemini returned a tool call with an empty name. Skipping.")
                    continue

                # Generate a deterministic ID for this tool call within the wrapper lifecycle
                self._tool_call_counter += 1
                tool_call_id = f"call_{self._tool_call_counter}"

                # Store function name for later retrieval when processing tool responses
                self._function_names[tool_call_id] = part.function_call.name

                # Store and serialize thought_signature if present (Gemini-specific metadata)
                # This is required by Gemini 3 for multi-turn function calling
                thought_sig_b64 = None
                if hasattr(part, 'thought_signature') and part.thought_signature:
                    self._thought_signatures[tool_call_id] = part.thought_signature
                    # Serialize as base64 for JSON storage/persistence
                    import base64
                    thought_sig_b64 = base64.b64encode(part.thought_signature).decode('ascii')
                    logger.debug(f"Stored thought_signature for tool_call {tool_call_id}")

                tool_call_dict = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(dict(part.function_call.args), default=str)
                    }
                }
                # Include thought_signature in serialized format for persistence
                if thought_sig_b64:
                    tool_call_dict["thought_signature"] = thought_sig_b64
                tool_calls.append(tool_call_dict)

        if text_parts:
            message["content"] = "\n".join(text_parts)
        else:
            message["content"] = ""

        if tool_calls:
            message["tool_calls"] = tool_calls
            # If we have tool calls, OpenAI often expects content to be None if it's empty
            if not message["content"]:
                message["content"] = None

        # Extract usage
        usage = response.usage_metadata
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) or 0 if usage else 0,
            "completion_tokens": getattr(usage, "candidates_token_count", None) or 0 if usage else 0,
            "total_tokens": getattr(usage, "total_token_count", None) or 0 if usage else 0
        }

        return {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop"
                }
            ],
            "usage": usage_dict,
            "model": self.model_name
        }
