from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict

from .errors import MCPUnavailableError, build_error_payload
from .models import MCPServerConfig

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    MCP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import behavior depends on environment
    ClientSession = None
    StdioServerParameters = None
    sse_client = None
    stdio_client = None
    streamable_http_client = None
    MCP_IMPORT_ERROR = exc


@dataclass
class MCPToolInfo:
    server_name: str
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPConnectionManager:
    def __init__(self, server_configs: Dict[str, MCPServerConfig]) -> None:
        self.server_configs = server_configs

    def _open_transport(self, config: MCPServerConfig):
        transport = (config.transport or "stdio").strip().lower()
        if MCP_IMPORT_ERROR is not None:
            raise MCPUnavailableError(f"MCP support is unavailable: {MCP_IMPORT_ERROR}")
        if transport == "stdio":
            server = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env or None,
                cwd=config.cwd,
            )
            return stdio_client(server)
        if transport == "sse":
            if not config.url:
                raise MCPUnavailableError("MCP SSE transport requires a url.")
            return sse_client(config.url)
        if transport == "streamable-http":
            if not config.url:
                raise MCPUnavailableError("MCP streamable-http transport requires a url.")
            return streamable_http_client(config.url)
        raise MCPUnavailableError(f"Unsupported MCP transport: {config.transport}")

    async def _create_session(self, server_name: str):
        config = self.server_configs.get(server_name)
        if config is None:
            raise MCPUnavailableError(f"MCP server '{server_name}' is not configured.")

        stack = AsyncExitStack()
        read_stream, write_stream = await stack.enter_async_context(self._open_transport(config))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return stack, session

    async def list_tools(self, server_name: str) -> Dict[str, MCPToolInfo]:
        stack, session = await self._create_session(server_name)
        try:
            tools_response = await session.list_tools()
            tools: Dict[str, MCPToolInfo] = {}
            for tool in getattr(tools_response, "tools", []) or []:
                tools[tool.name] = MCPToolInfo(
                    server_name=server_name,
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    input_schema=getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}},
                )
            return tools
        finally:
            await stack.aclose()

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            stack, session = await self._create_session(server_name)
            try:
                result = await session.call_tool(tool_name, arguments or {})
            finally:
                await stack.aclose()
            if hasattr(result, "model_dump"):
                dumped = result.model_dump()
            elif hasattr(result, "__dict__"):
                dumped = dict(result.__dict__)
            else:
                dumped = {"value": result}
            return {
                "ok": True,
                "result": dumped,
                "is_error": bool(getattr(result, "isError", False)),
                "trace_summary": f"MCP tool '{tool_name}' completed on server '{server_name}'.",
                "server": server_name,
                "tool_name": tool_name,
            }
        except Exception as exc:
            return build_error_payload(
                error="mcp_tool_failed",
                detail=str(exc),
                trace_summary=f"MCP tool '{tool_name}' failed on server '{server_name}'.",
                context={"server": server_name, "tool_name": tool_name},
            )

    async def aclose(self) -> None:
        return None
