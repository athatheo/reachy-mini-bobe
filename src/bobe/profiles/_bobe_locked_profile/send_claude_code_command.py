"""Profile tool that stages a Claude Code command for confirmation."""

from typing import Any

from bobe.tools.core_tools import Tool, ToolDependencies
from bobe.claude_code_session import get_claude_code_session_controller


class SendClaudeCodeCommand(Tool):
    """Stage a Claude Code command for exact spoken confirmation."""

    name = "send_claude_code_command"
    description = (
        "Use when the user asks you to tell Claude Code to do something. "
        "This tool only stages the command and returns the exact confirmation phrase; "
        "it does not send the command directly."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The exact instruction to send to Claude Code after confirmation.",
            },
        },
        "required": ["command"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Stage a command for confirmation."""
        command = str(kwargs.get("command") or "").strip()
        return get_claude_code_session_controller().request_send(command)
