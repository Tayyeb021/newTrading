"""MCP server: lets an AI assistant READ the trading system.

    python ops/mcp_server.py            # stdio transport, as Claude Code launches it

Registered in `.mcp.json` at the project root, so Claude Code in this
repository sees the tools automatically. Every tool is annotated read-only and
is implemented in `ops/mcp_tools.py`, which has no code path to any execution
write: no order can be placed, changed or closed through this server, and the
kill switch cannot be touched. An assistant here can explain; it cannot act.

The one tool that opens a connection, `live_account`, connects to the running
MetaTrader terminal, reads the account and positions, and disconnects.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from ops import mcp_tools  # noqa: E402

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

server = MCPServer(
    "trading",
    instructions=(
        "Read-only access to a systematic futures/CFD trading system: its live journal, "
        "session state, shadow-run digests, the pre-registered research verdicts and the "
        "research log. Nothing here can trade. If asked to place, change or close an order, "
        "or to engage or clear the kill switch, say that this server cannot and that the "
        "operator must do it directly."
    ),
)

server.tool(name="status", annotations=READ_ONLY)(mcp_tools.status)
server.tool(name="journal", annotations=READ_ONLY)(mcp_tools.journal)
server.tool(name="decisions", annotations=READ_ONLY)(mcp_tools.decisions)
server.tool(name="shadow_report", annotations=READ_ONLY)(mcp_tools.shadow_report)
server.tool(name="verdicts", annotations=READ_ONLY)(mcp_tools.verdicts)
server.tool(name="research_log", annotations=READ_ONLY)(mcp_tools.research_log)
server.tool(name="capital_ladder", annotations=READ_ONLY)(mcp_tools.capital_ladder)
server.tool(name="live_account", annotations=ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))(mcp_tools.live_account)


if __name__ == "__main__":
    server.run("stdio")
