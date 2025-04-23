"""Azure AI model Service MCP Server"""

import functools
import json
import logging
import re
import shutil
import subprocess
import sys
from typing import Callable, ParamSpec, TypeVar

import httpx
from jinja2.sandbox import SandboxedEnvironment
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("azure_model_mcp")


P = ParamSpec("P")
T = TypeVar("T")

# Initialize MCP and server
server_initialized = True
mcp = FastMCP(
    "azure-models",
    description="MCP server for Azure AI Model API integration",
    dependencies=["httpx", "azure-cli", "jinja2"],
)


def az(*args: str) -> dict:
    """Run azure-cli and return output

    :param str *args: The command line arguments to provide to git
    :returns: The standard output of the git command. Surrounding whitespace is removed
    :rtype: str
    """
    output = subprocess.run(
        [sys.executable, "-m", "azure.cli", *args, "-o", "json"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    return json.loads(output)


@mcp.tool()
def deploy_model(
    deployment_name: str,
    model_name: str,
    model_version: str,
    azure_ai_services_name: str,
    resource_group: str,
) -> None:
    """Deploy a model to azure ai

    Args:
        model_name: The name of the model to deploy
        model_version: The version of the model to deploy
        azure_ai_services_name: The name of cognitive services account to deploy to. This
            would be the name as it appears in the resource list on portal.azure.com
        resource_group: Then name of the resource group the cognitive services account is in.
    """
    az(
        "cognitiveservices",
        "account",
        "deployment",
        "create",
        "--model-format",
        "OpenAI",
        "--deployment-name",
        deployment_name,
        "--model-name",
        model_name,
        "--model-version",
        model_version,
        "--name",
        azure_ai_services_name,
        "--resource-group",
        resource_group,
    )


@mcp.tool()
def get_model_quotas(location: str) -> None:
    """Get model usages

    Args:
        location: The location to get usages in
    """
    return az("cognitiveservices", "usage", "list", "--location", location)


@mcp.tool()
def list_models(resource_group: str, azure_ai_services_name: str) -> str:
    """List available models for azure ai service

    Args:
        model_name: The name of the model to deploy
        model_version: The version of the model to deploy
        azure_ai_services_name: The name of cognitive services account to deploy to. This
            would be the name as it appears in the resource list on portal.azure.com
        resource_group: Then name of the resource group the cognitive services account is in.
    """

    return az(
        "cognitiveservices",
        "account",
        "list-models",
        "--resource-group",
        resource_group,
        "--name",
        azure_ai_services_name,
    )


@mcp.tool()
async def get_code_sample_for_deployment(
    deployment_name: str, resource_group: str, azure_ai_services_name: str
) -> str:
    """Get a code snippet for a deployment

    Args:
        deployment_name: The name of the deployment
        azure_ai_services_name: The name of cognitive services account to deploy to. This
            would be the name as it appears in the resource list on portal.azure.com
        resource_group: Then name of the resource group the cognitive services account is in.
    """
    account = az(
        "cognitiveservices",
        "account",
        "show",
        "--name",
        azure_ai_services_name,
        "--resource-group",
        resource_group,
    )
    endpoint = account["properties"]["endpoint"]

    deployment = az(
        "cognitiveservices",
        "account",
        "deployment",
        "show",
        "--name",
        azure_ai_services_name,
        "--resource-group",
        resource_group,
        "--deployment-name",
        deployment_name,
    )

    model_name = deployment["properties"]["model"]["name"]

    async with httpx.AsyncClient() as client:
        ejs_template = (
            await client.get(
                "https://ai.azure.com/modelcache/code2/oai-sdk-key-auth/en/chat-completion-python-template.md",
            )
        ).text

        model_template_config = (
            await client.get(
                f"https://ai.azure.com/modelcache/widgets/en/Serverless/azure-openai/{model_name}.json"
            )
        ).json()

    naive_jinja2_template = re.sub(r"<%=\s+([\w\.]+)\s%>", r"{{ \1|e }}", ejs_template)

    env = SandboxedEnvironment()

    template = env.from_string(naive_jinja2_template)

    example_content = [
        history["content"]
        for history in model_template_config[0]["config"]["examples"][0]["chatHistory"]
    ]

    return template.render(
        **{
            "endpointUrl": endpoint,
            "deploymentName": deployment_name,
            "modelName": model_name,
            "example": {
                "example_1": example_content[0],
                "example_2": example_content[1],
                "example_3": example_content[2],
            },
        }
    )


if __name__ == "__main__":
    status = (
        "successfully initialized" if server_initialized else "initialization failed"
    )
    print(
        f"\n{'=' * 50}\nAzure AI Models MCP Server {status}\nStarting server...\n{'=' * 50}\n"
    )
    mcp.run()
