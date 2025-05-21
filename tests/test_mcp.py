import os
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .utils import construct_openai_tools_from_mcp_tools, invoke_llm_with_tools
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_client_message_1():
    # In the args, we omit "--no-cache" to reduce latency
    server_params = StdioServerParameters(
        command="pipx",
        args=["run", "--spec", "..", "run-azure-ai-foundry-mcp"],
    )

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            openai_tools = construct_openai_tools_from_mcp_tools(
                mcp_tools=tools_response,
            )
            aoai_client = AzureOpenAI(
                azure_endpoint=os.environ["AOAI_ENDPOINT"],
                api_version=os.environ["AOAI_API_VERSION"],
                azure_ad_token_provider=token_provider,
            )

            completion = invoke_llm_with_tools(
                user_message="Tell me the Azure AI Foundry Labs projects",
                aoai_client=aoai_client,
                model=os.environ["AOAI_MODEL"],
                tools=openai_tools,
            )
            response_message = completion.choices[0].message

            # TODO dennis
            expected_tool_call_name = "list_azure_ai_foundry_labs_projects"
            actual_tool_calls = response_message.tool_calls
            assert len(actual_tool_calls) > 0
            assert isinstance(actual_tool_calls[0], ChatCompletionMessageToolCall)
            assert actual_tool_calls[0].function.name == expected_tool_call_name
