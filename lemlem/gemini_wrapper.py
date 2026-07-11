"""
Gemini API wrapper for lemlem.

Handles Gemini's native API format and automatic thought signature management.
Converts between OpenAI format (used internally by lemlem) and Gemini's native format.
"""

import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai
    from google.genai import types

logger = logging.getLogger(__name__)

# Matches "gemini-<major>[.<minor>]" so we can gate GA-3.5+ behavior (e.g. dropping
# temperature, which Gemini 3.5 no longer recommends).
_GEMINI_VERSION_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")


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
                        fc_kwargs = {"name": func_name, "args": args}
                        # GA Gemini 3.5 correlates function_call.id with function_response.id
                        if tool_call_id:
                            fc_kwargs["id"] = tool_call_id
                        part_kwargs = {"function_call": FunctionCall(**fc_kwargs)}
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

                # GA Gemini 3.5 requires FunctionResponse parts to carry id + matching name
                fr_kwargs = {"name": func_name, "response": result}
                if tool_call_id:
                    fr_kwargs["id"] = tool_call_id
                contents.append(
                    Content(
                        role="user",
                        parts=[Part(function_response=FunctionResponse(**fr_kwargs))]
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

    def _is_v35_or_newer(self) -> bool:
        """True for gemini-3.5 and later (GA models that no longer recommend temperature)."""
        match = _GEMINI_VERSION_RE.search(self.model_name or "")
        if not match:
            return False
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        return (major, minor) >= (3, 5)

    def _consume_function_call_part(self, part: Any) -> Optional[Dict[str, Any]]:
        """Convert one Gemini function_call part to an OpenAI tool_call dict.

        Shared by the streaming and non-streaming paths so both produce identical
        tool_call shapes (deterministic id, persisted base64 thought_signature).
        """
        import base64
        import json

        fc = getattr(part, "function_call", None)
        if not fc or not getattr(fc, "name", None):
            return None

        self._tool_call_counter += 1
        tool_call_id = f"call_{self._tool_call_counter}"
        self._function_names[tool_call_id] = fc.name

        thought_sig_b64 = None
        if getattr(part, "thought_signature", None):
            self._thought_signatures[tool_call_id] = part.thought_signature
            thought_sig_b64 = base64.b64encode(part.thought_signature).decode("ascii")

        tool_call_dict = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": fc.name,
                "arguments": json.dumps(dict(fc.args), default=str),
            },
        }
        if thought_sig_b64:
            tool_call_dict["thought_signature"] = thought_sig_b64
        return tool_call_dict

    def _build_openai_payload(
        self,
        text_parts: List[str],
        tool_calls: List[Dict[str, Any]],
        usage_meta: Any,
    ) -> Dict[str, Any]:
        """Assemble the OpenAI-style response dict from accumulated parts + usage."""
        message: Dict[str, Any] = {"role": "assistant"}
        message["content"] = "\n".join(text_parts) if text_parts else ""
        if tool_calls:
            message["tool_calls"] = tool_calls
            if not message["content"]:
                message["content"] = None

        usage_dict = {
            "prompt_tokens": getattr(usage_meta, "prompt_token_count", None) or 0 if usage_meta else 0,
            "completion_tokens": getattr(usage_meta, "candidates_token_count", None) or 0 if usage_meta else 0,
            "total_tokens": getattr(usage_meta, "total_token_count", None) or 0 if usage_meta else 0,
        }
        return {
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": usage_dict,
            "model": self.model_name,
        }

    def _assemble_streamed_response(
        self,
        text_parts: List[str],
        tool_calls: List[Dict[str, Any]],
        usage_meta: Any,
    ) -> Dict[str, Any]:
        """Build the final non-streaming-equivalent response after a stream completes."""
        return self._build_openai_payload(text_parts, tool_calls, usage_meta)

    def generate_content(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        *,
        on_stream_delta: Optional[Callable[[Dict[str, Any]], None]] = None,
        stream_iteration: Optional[int] = None,
        thinking_level: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini API with OpenAI-compatible interface.

        Args:
            messages: OpenAI-style messages
            tools: OpenAI-style tools
            temperature: Temperature for generation
            max_tokens: Max output tokens
            on_stream_delta: Optional callback fired with incremental text deltas. When
                provided, the call uses the streaming endpoint and still returns the
                fully-accumulated OpenAI-style dict (return contract unchanged).
            stream_iteration: Tool-loop iteration number, stamped onto stream deltas.
            thinking_level: Optional GA thinking_level enum (minimal/low/medium/high).
                Unset relies on the API default (medium for Gemini 3.5).
            **kwargs: Additional Gemini-specific params

        Returns:
            OpenAI-style response dict
        """

        # Convert to Gemini format
        contents = self.convert_openai_messages_to_gemini(messages)
        gemini_tools = self.convert_openai_tools_to_gemini(tools) if tools else None

        # Build config. GA Gemini 3.5+ no longer recommends temperature/top_p/top_k.
        config_params: Dict[str, Any] = {}
        if temperature is not None and not self._is_v35_or_newer():
            config_params["temperature"] = temperature
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens
        if gemini_tools:
            config_params["tools"] = gemini_tools

        # Call Gemini API
        try:
            GenerateContentConfig = self._types.GenerateContentConfig
            # Extract timeout from kwargs if present, as it's passed to the client method, not the config
            request_timeout = kwargs.pop("timeout", None)

            config = GenerateContentConfig(**config_params) if config_params else GenerateContentConfig()

            # Optional GA thinking_level. Applied defensively so older SDKs degrade gracefully.
            if thinking_level:
                try:
                    config.thinking_config = self._types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                except Exception as exc:
                    logger.warning("Could not set thinking_level=%s: %s", thinking_level, exc)

            call_kwargs = {
                "model": self.model_name,
                "contents": contents,
                "config": config,
            }

            if request_timeout is not None:
                # google-genai expects timeout in milliseconds via types.HttpOptions
                # Convert from seconds to milliseconds, with minimum of 10s (10000ms) per API requirements
                timeout_ms = max(10000, int(request_timeout * 1000))
                HttpOptions = self._types.HttpOptions
                call_kwargs["config"].http_options = HttpOptions(timeout=timeout_ms)

            hard_timeout = None
            if request_timeout is not None:
                try:
                    hard_timeout = max(10.0, float(request_timeout) + 15.0)
                except (TypeError, ValueError):
                    hard_timeout = None

            if on_stream_delta is not None:
                return self._generate_content_streaming(
                    call_kwargs, on_stream_delta, stream_iteration, hard_timeout
                )

            response = self.client.models.generate_content(**call_kwargs)

            # Convert response to OpenAI format
            return self._convert_gemini_response_to_openai(response)

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

    def _generate_content_streaming(
        self,
        call_kwargs: Dict[str, Any],
        on_stream_delta: Callable[[Dict[str, Any]], None],
        stream_iteration: Optional[int],
        hard_timeout: Optional[float],
    ) -> Dict[str, Any]:
        """Stream a Gemini response, emitting debounced text deltas, and return the
        fully-accumulated OpenAI-style dict (identical to the non-streaming path)."""
        flush_chars = int(os.getenv("LEMLEM_STREAM_DELTA_CHARS", "280"))
        flush_interval = float(os.getenv("LEMLEM_STREAM_DELTA_INTERVAL_S", "0.4"))

        # Streamed text arrives as fragments of one continuous message; concatenate them
        # (no separator) so the assembled content matches a non-streamed single-part response.
        text_segments: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage_meta: Any = None

        buffer: List[str] = []
        chars_since = 0
        last_emit = time.monotonic()
        deadline = (time.monotonic() + hard_timeout) if hard_timeout else None

        def _emit(done: bool) -> None:
            nonlocal buffer, chars_since, last_emit
            delta_text = "".join(buffer)
            if not delta_text and not done:
                return
            try:
                on_stream_delta(
                    {
                        "iteration": stream_iteration,
                        "call_id": None,
                        "delta_text": delta_text,
                        "done": done,
                    }
                )
            except Exception:
                logger.debug("on_stream_delta callback failed", exc_info=True)
            buffer = []
            chars_since = 0
            last_emit = time.monotonic()

        stream = self.client.models.generate_content_stream(**call_kwargs)
        try:
            for chunk in stream:
                if deadline and time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Gemini generate_content stream timed out after {hard_timeout:.1f}s"
                    )
                if getattr(chunk, "usage_metadata", None):
                    usage_meta = chunk.usage_metadata
                candidates = getattr(chunk, "candidates", None)
                if not candidates:
                    continue
                content = candidates[0].content
                if not content or not content.parts:
                    continue
                for part in content.parts:
                    if getattr(part, "text", None):
                        text_segments.append(part.text)
                        buffer.append(part.text)
                        chars_since += len(part.text)
                    elif getattr(part, "function_call", None):
                        tool_call = self._consume_function_call_part(part)
                        if tool_call:
                            tool_calls.append(tool_call)
                now = time.monotonic()
                if chars_since >= flush_chars or (now - last_emit) >= flush_interval:
                    _emit(done=False)
        finally:
            # Always emit a final delta so the UI clears the transient bubble.
            _emit(done=True)

        text_parts = ["".join(text_segments)] if text_segments else []
        return self._assemble_streamed_response(text_parts, tool_calls, usage_meta)

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

        # Safety check: ensure parts exists
        if not content.parts:
            logger.warning(f"Gemini response has no parts. Candidate finish_reason: {getattr(candidate, 'finish_reason', 'unknown')}, content: {content}")
            return self._build_openai_payload([], [], response.usage_metadata)

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for part in content.parts:
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                tool_call = self._consume_function_call_part(part)
                if tool_call:
                    tool_calls.append(tool_call)

        return self._build_openai_payload(text_parts, tool_calls, response.usage_metadata)
