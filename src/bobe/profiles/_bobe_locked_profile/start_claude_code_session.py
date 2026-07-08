"""Profile tool that starts a managed Claude Code session."""

from typing import Any

from bobe.tools.core_tools import Tool, ToolDependencies
from bobe.claude_code_session import get_claude_code_session_controller


class StartClaudeCodeSession(Tool):
    """Start or reuse a Mac-daemon-managed Claude Code session."""

    name = "start_claude_code_session"
    description = "Start or reuse the managed Claude Code session on the Mac mini."
    parameters_schema = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Start or reuse the managed session."""
        return await get_claude_code_session_controller().start()
