"""
Utilities for helping run tests.
"""

from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from mcp.types import ListToolsResult


def construct_openai_tools_from_mcp_tools(mcp_tools: ListToolsResult) -> list[dict]:
    """
    Given a tools list from MCP server, convert it to the format required to feed into Azure OpenAI chat completion.
    """
    final_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools.tools
    ]
    return final_tools


def invoke_llm_with_tools(
    user_message: str,
    tools: list[dict],
    aoai_client: AzureOpenAI,
    model: str,
) -> ChatCompletion:
    """
    Invoke a single LLM inference step on a user message, including the specified tools, and return the response
    """
    messages = [
        dict(
            role="user",
            message=user_message,
        )
    ]

    completion = aoai_client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return completion
