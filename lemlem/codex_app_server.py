from __future__ import annotations

import asyncio
import inspect
import json
import os
import select
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence


class CodexAppServerError(RuntimeError):
    pass


class _ProtocolClient:
    def __init__(self, timeout: float = 1800.0) -> None:
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self.next_id = 1

    def __enter__(self) -> "_ProtocolClient":
        self.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "CODEX_HOME",
                "HOME",
                "PATH",
                "LANG",
                "LC_ALL",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "NO_PROXY",
            }
        }
        self.process = subprocess.Popen(
            ["codex", "app-server", "--listen", "stdio://", "--strict-config"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "evergreen-lemlem",
                    "title": "Evergreen",
                    "version": "1.0.0",
                },
                "capabilities": {"experimentalApi": False},
            },
        )
        self.notify("initialized", {})

    def close(self) -> None:
        process = self.process
        self.process = None
        if not process:
            return
        if process.stdin:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout:
            process.stdout.close()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"method": method, "params": params})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        self._write({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + min(self.timeout, 30.0)
        while True:
            message = self._read(deadline)
            if message.get("id") != request_id or "method" in message:
                continue
            if "error" in message:
                error = message.get("error") or {}
                raise CodexAppServerError(str(error.get("message") or "request failed"))
            result = message.get("result")
            if not isinstance(result, dict):
                raise CodexAppServerError(f"Codex request '{method}' returned no object")
            return result

    def list_models(self) -> list[dict[str, Any]]:
        result = self.request("model/list", {"includeHidden": True})
        return [item for item in result.get("data", []) if isinstance(item, dict)]

    def validate_model(self, model: str, effort: str) -> None:
        for item in self.list_models():
            if model not in {item.get("model"), item.get("id")}:
                continue
            efforts = {
                entry.get("reasoningEffort") if isinstance(entry, dict) else entry
                for entry in item.get("supportedReasoningEfforts", [])
            }
            if effort not in efforts:
                raise CodexAppServerError(
                    f"Model '{model}' does not support effort '{effort}'"
                )
            return
        raise CodexAppServerError(f"Model '{model}' is not advertised by Codex")

    def start_thread(self, cwd: Path, model: str) -> str:
        result = self.request(
            "thread/start",
            {
                "model": model,
                "cwd": str(cwd),
                "approvalPolicy": "never",
                "sandbox": "workspace-write",
                "ephemeral": True,
            },
        )
        thread_id = str((result.get("thread") or {}).get("id") or "")
        if not thread_id:
            raise CodexAppServerError("thread/start returned no thread id")
        return thread_id

    def turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        cwd: Path,
        model: str,
        effort: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "model": model,
                "effort": effort,
                "cwd": str(cwd),
                "approvalPolicy": "never",
                "sandboxPolicy": {
                    "type": "workspaceWrite",
                    "writableRoots": [str(cwd)],
                },
                "outputSchema": output_schema,
            },
        )
        turn_id = str((result.get("turn") or {}).get("id") or "")
        if not turn_id:
            raise CodexAppServerError("turn/start returned no turn id")
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        deadline = time.monotonic() + self.timeout
        while True:
            message = self._read(deadline)
            method = message.get("method")
            params = message.get("params") or {}
            message_turn_id = str(
                params.get("turnId") or (params.get("turn") or {}).get("id") or ""
            )
            if message_turn_id != turn_id:
                continue
            if method == "thread/tokenUsage/updated":
                usage = dict((params.get("tokenUsage") or {}).get("last") or {})
            elif method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage" and item.get("text"):
                    text_parts.append(str(item["text"]))
            elif method == "turn/completed":
                turn = params.get("turn") or {}
                status = str(turn.get("status") or "failed")
                if status != "completed":
                    raise CodexAppServerError(
                        f"Codex turn ended with status={status}: {turn.get('error')}"
                    )
                return {
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "final_text": "\n".join(text_parts).strip(),
                    "usage": {
                        "input_tokens": int(usage.get("inputTokens") or 0),
                        "cached_input_tokens": int(usage.get("cachedInputTokens") or 0),
                        "output_tokens": int(usage.get("outputTokens") or 0),
                        "reasoning_output_tokens": int(
                            usage.get("reasoningOutputTokens") or 0
                        ),
                        "total_tokens": int(usage.get("totalTokens") or 0),
                        "complete": bool(usage),
                    },
                }

    def _write(self, payload: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise CodexAppServerError("Codex App Server is not running")
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def _read(self, deadline: float) -> dict[str, Any]:
        if not self.process or not self.process.stdout:
            raise CodexAppServerError("Codex App Server is not running")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodexAppServerError("Codex App Server timed out")
        readable, _, _ = select.select([self.process.stdout], [], [], remaining)
        if not readable:
            raise CodexAppServerError("Codex App Server timed out")
        raw = self.process.stdout.readline()
        if not raw:
            raise CodexAppServerError("Codex App Server exited unexpectedly")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CodexAppServerError("Codex App Server returned malformed JSON") from exc
        if not isinstance(value, dict):
            raise CodexAppServerError("Codex App Server returned a non-object message")
        return value


class CodexAppServerRuntime:
    def call_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        model: str,
        effort: str,
        cwd: str | Path,
        output_schema: dict[str, Any] | None = None,
        tools: Sequence[Any] | None = None,
        max_tool_iterations: int = 6,
        on_turn: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        project_dir = Path(cwd).resolve()
        interactions: list[dict[str, Any]] = []
        total_usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
            "complete": False,
        }
        registry = {str(tool.name): tool for tool in tools or []}
        if registry:
            definitions = [self._tool_definition(tool) for tool in registry.values()]
            prompt = (
                system_prompt.rstrip()
                + "\n\nTOOL_PROTOCOL: Return exactly one JSON object containing either "
                "`tool_call` with name and arguments, or `final` with the final response. "
                "Never claim a tool ran without its result.\nAvailable tools:\n"
                + json.dumps(definitions, ensure_ascii=False, default=str)
                + "\nFinal output schema:\n"
                + json.dumps(output_schema or {"type": "object"})
                + "\nInput JSON:\n"
                + json.dumps(user_payload, ensure_ascii=False, default=str)
            )
            schema = {"type": "object", "additionalProperties": True}
        else:
            prompt = (
                system_prompt.rstrip()
                + "\n\nInput JSON:\n"
                + json.dumps(user_payload, ensure_ascii=False, default=str)
            )
            schema = output_schema or {"type": "object", "additionalProperties": True}
        with _ProtocolClient() as client:
            client.validate_model(model, effort)
            thread_id = client.start_thread(project_dir, model)
            for iteration in range(1, max_tool_iterations + 2):
                turn = client.turn(
                    thread_id=thread_id,
                    prompt=prompt,
                    cwd=project_dir,
                    model=model,
                    effort=effort,
                    output_schema=schema,
                )
                for key in total_usage:
                    if key == "complete":
                        total_usage[key] = total_usage[key] or turn["usage"][key]
                    else:
                        total_usage[key] += turn["usage"][key]
                try:
                    payload = json.loads(turn["final_text"])
                except json.JSONDecodeError as exc:
                    raise CodexAppServerError("Codex response was not valid JSON") from exc
                if not isinstance(payload, dict):
                    raise CodexAppServerError("Codex response was not a JSON object")
                if not registry:
                    final = payload
                else:
                    final = payload.get("final")
                if isinstance(final, dict):
                    return {
                        "data": final,
                        "final_text": json.dumps(final, ensure_ascii=False),
                        "usage": total_usage,
                        "thread_id": thread_id,
                        "turn_id": turn["turn_id"],
                        "tool_interactions": interactions,
                    }
                call = payload.get("tool_call")
                if not isinstance(call, dict) or iteration > max_tool_iterations:
                    raise CodexAppServerError("Codex tool loop ended without final output")
                name = str(call.get("name") or "")
                arguments = call.get("arguments")
                if name not in registry or not isinstance(arguments, dict):
                    raise CodexAppServerError(f"Codex requested invalid tool '{name}'")
                call_id = str(uuid.uuid4())
                if on_turn:
                    on_turn(
                        {
                            "type": "tool_call",
                            "iteration": iteration,
                            "metadata": {
                                "tool_name": name,
                                "call_id": call_id,
                                "arguments": arguments,
                            },
                        }
                    )
                try:
                    output = self._execute(registry[name], arguments)
                except Exception as exc:
                    output = {"is_error": True, "error": str(exc)}
                interaction = {
                    "iteration": iteration,
                    "tool_name": name,
                    "call_id": call_id,
                    "arguments": arguments,
                    "output": output,
                }
                interactions.append(interaction)
                if on_turn:
                    on_turn(
                        {
                            "type": "tool_result",
                            "iteration": iteration,
                            "content": output,
                            "metadata": interaction,
                        }
                    )
                prompt = (
                    f"Tool result for {name} (call_id={call_id}):\n"
                    + json.dumps(output, ensure_ascii=False, default=str)
                    + "\nContinue using TOOL_PROTOCOL."
                )
        raise CodexAppServerError("Codex tool loop ended unexpectedly")

    @staticmethod
    def _tool_definition(tool: Any) -> dict[str, Any]:
        schema = getattr(tool, "input_schema", None) or getattr(tool, "parameters", None)
        if not isinstance(schema, dict):
            schema = {"type": "object", "additionalProperties": True}
        return {
            "name": str(tool.name),
            "description": str(getattr(tool, "description", "")),
            "input_schema": schema,
        }

    @staticmethod
    def _execute(tool: Any, arguments: dict[str, Any]) -> Any:
        handler = getattr(tool, "handler", None)
        if callable(handler):
            result = handler(arguments)
        else:
            function = getattr(tool, "function", None)
            if not callable(function):
                raise TypeError(f"Tool '{tool.name}' has no handler")
            result = function(**arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result
