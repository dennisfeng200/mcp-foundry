from typing import Any
from mcp.types import ListToolsResult
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_client_lists_tools():
    server_params = StdioServerParameters(
        command="pipx",
        args=["run", "--no-cache", "--spec", "..", "run-azure-ai-foundry-mcp"],
    )

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            response = await session.list_tools()
            tools = response.tools
            assert tools, "Expected at least one tool from the MCP server"
