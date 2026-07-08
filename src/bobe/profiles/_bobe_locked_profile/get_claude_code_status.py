"""Profile tool that reads managed Claude Code session status."""

from typing import Any

from bobe.tools.core_tools import Tool, ToolDependencies
from bobe.claude_code_session import get_claude_code_session_controller


class GetClaudeCodeStatus(Tool):
    """Fetch status for the Mac-daemon-managed Claude Code session."""

    name = "get_claude_code_status"
    description = "Get the current managed Claude Code session status and latest result."
    parameters_schema = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Fetch managed session status."""
        return await get_claude_code_session_controller().status()
