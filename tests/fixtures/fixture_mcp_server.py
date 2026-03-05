from mcp.server.fastmcp import FastMCP


mcp = FastMCP("fixturemcp")


@mcp.tool()
def add_numbers(a: int, b: int) -> dict:
    return {"sum": a + b}


if __name__ == "__main__":
    mcp.run()
