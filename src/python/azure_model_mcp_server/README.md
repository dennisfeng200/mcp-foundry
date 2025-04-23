# Azure AI Model MCP Server

This MCP (Model Context Protocol) server integrates with Azure OpenAI services to interact with models.


## Features

- Deploy Azure OpenAI models to your Cognitive Services account
- Check available quotas for models in specific Azure regions
- List available models
- Generate code samples for interacting with deployed models



## Usage


### Configuration with Claude Desktop or Other MCP Clients

Add this to your MCP client configuration:

```json
{
  "mcpServers": {
    "azure-ai-models": {
        "command": "uv",
        "args": [
            "run",
            "--prerelease=allow",
            "--directory",
            "/path/to/src/python/azure_model_mcp_server/",
            "__main__.py"
        ]
    }
  }
}
```
