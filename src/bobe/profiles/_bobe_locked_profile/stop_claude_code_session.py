"""Profile tool that stops a managed Claude Code session."""

from typing import Any

from bobe.tools.core_tools import Tool, ToolDependencies
from bobe.claude_code_session import get_claude_code_session_controller


class StopClaudeCodeSession(Tool):
    """Stop the Mac-daemon-managed Claude Code session."""

    name = "stop_claude_code_session"
    description = "Stop the managed Claude Code session on the Mac mini."
    parameters_schema = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Stop the managed session."""
        return await get_claude_code_session_controller().stop()
