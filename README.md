# MCP Foundry Server

A Model Context Protocol server for Azure AI Foundry, providing a unified set of tools for knowledge, models, content safety, observability, and more.

[![GitHub watchers](https://img.shields.io/github/watchers/azure-ai-foundry/mcp-foundry.svg?style=social&label=Watch)](https://github.com/azure-ai-foundry/mcp-foundry/watchers)
[![GitHub forks](https://img.shields.io/github/forks/azure-ai-foundry/mcp-foundry.svg?style=social&label=Fork)](https://github.com/azure-ai-foundry/mcp-foundry/fork)
[![GitHub stars](https://img.shields.io/github/stars/azure-ai-foundry/mcp-foundry?style=social&label=Star)](https://github.com/azure-ai-foundry/mcp-foundry/stargazers)

[![Azure AI Community Discord](https://dcbadge.vercel.app/api/server/ByRwuEEgH4)](https://discord.gg/REmjGvvFpW)

## Tool Categories

### Foundry-Knowledge
#### Tools
- Tool 1: [placeholder]
- Tool 2: [placeholder]

### Foundry-Models
#### Tools
- Tool 1: [placeholder]
- Tool 2: [placeholder]

### Foundry-Content-Safety
#### Tools
- Tool 1: [placeholder]
- Tool 2: [placeholder]

### Foundry-Observability
#### Tools
- Tool 1: [placeholder]
- Tool 2: [placeholder]

---

## Configuration

Edit `mcp.json` to configure server options and tool settings.

## Deployment

### Quick Start

[![Install in VS Code](https://img.shields.io/static/v1?style=for-the-badge&label=Install+in+VS+Code&message=Open&color=007ACC&logo=visualstudiocode)](https://insiders.vscode.dev/redirect/mcp/install?name=Azure%20Foundry%20MCP%20Server&config=%7B%22command%22%3A%22pipx%22%2C%22args%22%3A%5B%22run%22%2C%22--no-cache%22%2C%22--spec%22%2C%22git%2Bhttps%3A%2F%2Fgithub.com%2Fazure-ai-foundry%2Fmcp-foundry.git%22%2C%22run-azure-ai-foundry-mcp%22%5D%7D)
[![Use The Template](https://img.shields.io/static/v1?style=for-the-badge&label=Use+The+Template&message=GitHub&color=181717&logo=github)](https://github.com/azure-ai-foundry/foundry-models-playground/generate)



---

### Manual Setup

#### 1. Install Python and pipx

Make sure you have **Python** installed along with **pipx**.  
Most modern Python installations include `pipx`, but if you don’t have it, you can install it with:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

---

#### 2. Configure Your MCP Client

Open the **MCP settings** in your client of choice.  
Follow the appropriate link below for detailed instructions:

- [Visual Studio Code – Copilot Chat](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [Claude](https://modelcontextprotocol.io/quickstart/user)
- [Cursor](https://docs.cursor.com/context/model-context-protocol)

---

#### 3. Add MCP Entry

Copy and paste the following JSON block into your MCP client’s configuration:

```json
"MCP Server For Azure AI Foundry": {
  "command": "pipx",
  "args": [
    "run",
    "--no-cache"
    "--spec",
    "git+https://github.com/azure-ai-foundry/mcp-foundry.git@msbuild2025",
    "run-azure-ai-foundry-mcp"
  ]
}
```

> This will automatically install and run the MCP server using `pipx`.

---

## License

MIT License. See LICENSE for details.
