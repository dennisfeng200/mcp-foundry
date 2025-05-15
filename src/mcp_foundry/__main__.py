from mcp.server.fastmcp import FastMCP, Context
import requests
import os
from dotenv import load_dotenv
import logging

from .mcp_foundry_model.models import ModelDetails
from .mcp_foundry_model.utils import get_client_headers_info, get_models_list, get_code_sample_for_github_model, get_code_sample_for_labs_model, get_code_sample_for_deployment_under_ai_services, get_ai_services_usage_list

load_dotenv()

# Configure logger
logger = logging.getLogger("mcp_foundry")
logging.basicConfig(level=logging.DEBUG)

mcp = FastMCP("azure-ai-foundry-mcp-server")
labs_api_url = os.environ.get("LABS_API_URL", "https://labs-mcp-api.azurewebsites.net/api/v1")


@mcp.tool()
async def list_models_from_model_catalog(ctx: Context, search_for_free_playground: bool = False, publisher_name = "", license_name = "") -> str:
    """
    Retrieves a list of supported models from the Azure AI Foundry catalog.

    This function is useful when a user requests a list of available Foundry models or Foundry Labs projects.
    It fetches models based on optional filters like whether the model supports free playground usage,
    the publisher name, and the license type. The function will return the list of models with useful fields.

    Parameters:
        ctx (Context): The context of the current session. Contains metadata about the request and session.
        search_for_free_playground (bool, optional): If `True`, filters models to include only those that
            can be used for free by users for prototyping. If `False`, all models will be included regardless of free playground support.
            Defaults to `False`.
        publisher_name (str, optional): A filter to specify the publisher of the models to retrieve. If provided,
            only models from this publisher will be returned. Defaults to an empty string, meaning no filter is applied.
        license_name (str, optional): A filter to specify the license type of the models to retrieve. If provided,
            only models with this license will be returned. Defaults to an empty string, meaning no filter is applied.

    Returns:
        str: A JSON-encoded string containing the list of models and their metadata. The list will include 
             model names, inference model names, summaries, and the total count of models retrieved.

    Usage:
        Use this function when users inquire about available models from the Azure AI Foundry catalog.
        It can also be used when filtering models by free playground usage, publisher name, or license type.
        If user didn't specify free playground or ask for models that support GitHub token, always explain that by default it will show the all the models but some of them would support free playground.
        Explain to the user that if they want to find models suitable for prototyping and free to use with support for free playground, they can look for models that supports free playground, or look for models that they can use with GitHub token.
    """
    max_pages = 5
    # Note: if max_pages becomes larger, the agent will find it more difficult to "summarize" the result, which may not be desired.
    # max_pages = 10

    logger.debug("Calling get_models_list with parameters:")
    logger.debug(f"search_for_free_playground: {search_for_free_playground}")
    logger.debug(f"publisher_name: {publisher_name}")
    logger.debug(f"license_name: {license_name}")
    logger.debug(f"max_pages: {max_pages}")

    models_list = get_models_list(ctx, search_for_free_playground, publisher_name, license_name, max_pages)

    return models_list.json()

@mcp.tool()
async def list_azure_ai_foundry_labs_projects(ctx: Context):
    """
    Retrieves a list of state-of-the-art AI models from Microsoft Research available in Azure AI Foundry Labs.

    This function is used when a user requests information about the cutting-edge models and projects developed by Microsoft Research within the Azure AI Foundry Labs. These models represent the latest advancements in AI research and are often experimental or in early development stages.

    Parameters:
        ctx (Context): The context of the current session, which includes metadata and session-specific information.

    Returns:
        list: A list containing the list of available AI models and projects in Azure AI Foundry Labs. The list will include information such as project names, descriptions, and possibly other metadata relevant to the state-of-the-art models.

    Usage:
        Use this function when a user wants to explore the latest models and research projects available in the Azure AI Foundry Labs. These projects are typically cutting-edge and may involve new or experimental features not yet widely available.

    Notes:
        - The models and projects in Azure AI Foundry Labs are generally from the forefront of AI research and may have specific requirements or experimental capabilities.
        - The list returned may change frequently as new models and projects are developed and made available for exploration.
    """

    headers = get_client_headers_info(ctx)

    response = requests.get(f"{labs_api_url}/projects?source=afl", headers=headers)
    if response.status_code != 200:
        return f"Error fetching projects from API: {response.status_code}"

    project_response = response.json()

    return project_response["projects"]

@mcp.tool()
async def list_deployments_from_azure_ai_services(ctx: Context):
    """
    Retrieves a list of deployments from Azure AI Services.

    This function is used when a user requests information about the available deployments in Azure AI Services. It provides an overview of the models and services that are currently deployed and available for use.

    Parameters:
        ctx (Context): The context of the current session, which includes metadata and session-specific information.

    Returns:
        list: A list containing the details of the deployments in Azure AI Services. The list will include information such as deployment names, descriptions, and possibly other metadata relevant to the deployed services.

    Usage:
        Use this function when a user wants to explore the available deployments in Azure AI Services. This can help users understand what models and services are currently operational and how they can be utilized.

    Notes:
        - The deployments listed may include various models and services that are part of Azure AI Services.
        - The list may change frequently as new deployments are added or existing ones are updated.
    """

    headers = get_client_headers_info(ctx)

    pass

@mcp.tool()
async def get_model_details_and_code_samples(model_name: str, ctx: Context):
    """
    Retrieves detailed information for a specific model from the Azure AI Foundry catalog.

    This function is used when a user requests detailed information about a particular model in the Foundry catalog.
    It fetches the model's metadata, capabilities, descriptions, and other relevant details associated with the given asset ID.

    Parameters:
        model_name (str): The name of the model whose details are to be retrieved. This is a required parameter.
        ctx (Context): The context of the current session, containing metadata about the request and session.

    Returns:
        dict: A dictionary containing the model's detailed information, including:
            - model name, version, framework, tags, datasets
            - model URL and storage location
            - model capabilities (e.g., agents, assistants, reasoning, tool-calling)
            - description, summary, and key capabilities
            - publisher information, licensing details, and terms of use
            - model creation and modification times
            - variant information, model metadata, and system requirements

    Usage:
        Call this function when you need to retrieve detailed information about a model using its asset ID. 
        This is useful when users inquire about a model's features, or when specific metadata about a model is required.
    """
    headers = get_client_headers_info(ctx)

    model_details = {
        "details": {},
        "code_sample_azure": None,
        "code_sample_github": None,
        "type": None
    }

    response = requests.get(f"{labs_api_url}/projects?source=afl", headers=headers)
    if response.status_code != 200:
        return f"Error fetching projects from API: {response.status_code}"

    project_response = response.json()

    project_names = [project["name"] for project in project_response["projects"]]

    if model_name in project_names:
        model_details["details"] = project_response["projects"][project_names.index(model_name)]
        model_details["code_sample"] = await get_code_sample_for_labs_model(model_name, ctx)
        model_details["type"] = "Labs"
        return ModelDetails(**model_details)
    
    model_list_details = get_models_list(ctx, model_name=model_name)
    if model_list_details.fetched_models_count == 0:
        return f"Model '{model_name}' not found in the catalog."
    
    model_list_details  = model_list_details.summaries[0]

    response = requests.get(f"https://ai.azure.com/api/westus2/modelregistry/v1.0/registry/models?assetIdOrReference={model_list_details['assetId']}", headers=headers)
    if response.status_code != 200:
        return f"Error fetching model details from API: {response.status_code}"

    model_details["details"] = response.json()

    # Free playground model add GH guidance to model details
    if "freePlayground" in model_details['details']['kvTags'] and model_details['details']['kvTags']["freePlayground"] == "true":
        model_details["type"] = "Free Playground"
        model_details["code_sample_github"] = await get_code_sample_for_github_model(model_list_details["publisher"], model_list_details["name"], ctx)

    # OpenAI model add OpenAI guidance to model details
    if model_list_details["deployment_options"]["openai"]:
        if not model_details["type"] == "Free Playground":
            model_details["type"] = "OpenAI"
        model_details["code_sample_azure"] = await get_code_sample_for_deployment_under_ai_services()

    # PayGo model add PayGo guidance to model details
    elif model_list_details["deployment_options"]["serverless_endpoint"]:
        model_details["type"] = "Serverless Endpoint"
        model_details["code_sample_azure"] = await get_code_sample_for_deployment_under_ai_services()

    # Managed compute model add managed compute guidance to model details
    elif model_list_details["deployment_options"]["managed_compute"]:
        model_details["type"] = "Managed Compute"
        pass

    return ModelDetails(**model_details)

@mcp.tool()
async def deploy_model_on_ai_services() -> str:
    """
    Deploys a model on Azure AI Services.

    This function is used to deploy a model on Azure AI Services, allowing users to integrate the model into their applications and utilize its capabilities.

    Returns:
        str: A string indicating the status of the deployment process.
    """

    pass

@mcp.tool()
def get_prototyping_instructions_for_github_and_labs(ctx: Context) -> str:
    """
    Provides comprehensive instructions and setup guidance for starting to work with Foundry models from Azure AI Foundry Labs.

    This function is crucial to call whenever a user begins talking about or expressing an interest in working with Foundry models. It provides the essential prototyping instructions that include setup, configuration, and the first steps in querying and utilizing the models. It should always be invoked before any other interactions with the models to ensure that the user has the necessary context and knowledge to proceed effectively.

    The instructions include:
        - Required setup for working with Foundry models.
        - Details about how to configure the environment.
        - How to query the models.
        - Best practices for using Foundry models in prototyping.
    
    Parameters:
        ctx (Context): The context of the current session, which may include session-specific information and metadata that can be used to customize the returned instructions.

    Returns:
        str: A detailed set of instructions to guide the user in setting up and using Foundry models, including steps on how to get started with queries and the prototyping process.

    Usage:
        Call this function at the beginning of any interaction involving Foundry models to provide the user with the necessary setup information and best practices. This ensures that the user can begin their work with all the foundational knowledge and tools needed.

    Notes:
        - This function should be the first step before any interaction with the Foundry models to ensure proper setup and understanding.
        - It is essential to invoke this function as it provides the groundwork for a successful prototyping experience with Foundry models.

    Importance:
        The function is critical for preparing the user to effectively use the Azure AI Foundry models, ensuring they have the proper guidance on how to interact with them from the very beginning.
    """

    headers = get_client_headers_info(ctx)
    response = requests.get(f"{labs_api_url}/resources/resource/copilot-instructions.md", headers=headers)
    if response.status_code != 200:
        return f"Error fetching instructions from API: {response.status_code}"

    copilot_instructions = response.json()
    return copilot_instructions["resource"]


def main() -> None:
    """Runs the MCP server"""
    print("Starting MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
