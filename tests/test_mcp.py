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


async def verify_mcp_tool_call(user_message: str, expected_tool_call_name: str, no_cache: bool = False):
    """
    Helper function to test MCP tool calling functionality with different messages and expected tools.

    Args:
        user_message: The query to send to the model
        expected_tool_call_name: The name of the tool we expect the model to call
        no_cache: Whether to use --no-cache flag when running the MCP server
    """
    args = ["run", "--spec", "..", "run-azure-ai-foundry-mcp"]
    if no_cache:
        args.insert(1, "--no-cache")

    server_params = StdioServerParameters(
        command="pipx",
        args=args,
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
                user_message=user_message,
                aoai_client=aoai_client,
                model=os.environ["AOAI_MODEL"],
                tools=openai_tools,
            )
            response_message = completion.choices[0].message

            actual_tool_calls = response_message.tool_calls
            assert len(actual_tool_calls) > 0, "Expected at least one tool call but got none"
            assert isinstance(actual_tool_calls[0], ChatCompletionMessageToolCall)
            assert actual_tool_calls[0].function.name == expected_tool_call_name

            return completion


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_client_message_1():
    """Test that the model correctly calls list_azure_ai_foundry_labs_projects tool"""
    await verify_mcp_tool_call(
        user_message="Tell me the Azure AI Foundry Labs projects",
        expected_tool_call_name="list_azure_ai_foundry_labs_projects",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_client_message_2():
    await verify_mcp_tool_call(
        user_message="I want to prototype an app with Azure AI Foundry Labs. Where do I start?",
        expected_tool_call_name="get_prototyping_instructions_for_github_and_labs",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_client_message_3():
    # TODO: Create a more sophisticated tool call verification step that handles the stochosticity.
    # This is because this pytest sometimes gives different tool calls which is expected of a vague freeform input
    await verify_mcp_tool_call(
        user_message="I want to use the Aurora model from Azure AI Foundry Labs; fetch details on how to implement it.",
        expected_tool_call_name="list_azure_ai_foundry_labs_projects",
    )
