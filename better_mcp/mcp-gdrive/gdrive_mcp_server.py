import logging
from mcp.server.fastmcp import FastMCP
from drive_core import search_files, get_metadata, get_text

logging.basicConfig(level=logging.INFO)
mcp = FastMCP("gdrive")

@mcp.tool()
def search_files_tool(query: str, page_size: int = 25):
    return search_files(query, page_size)

@mcp.tool()
def get_metadata_tool(file_id: str):
    return get_metadata(file_id)

@mcp.tool()
def get_text_tool(file_id: str, max_chars: int = 200_000):
    return get_text(file_id, max_chars)

@mcp.resource("gdrive://file/{file_id}")
def read_resource(file_id: str):
    return get_text(file_id)

if __name__ == "__main__":
    mcp.run(transport="stdio")
