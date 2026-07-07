"""Profile tool that starts a confirmed Claude Code launch request."""

from typing import Any

from bobe.tools.core_tools import Tool, ToolDependencies
from bobe.claude_code_launch import get_claude_code_launch_controller


class LaunchClaudeCode(Tool):
    """Create or cancel a pending Claude Code launch request."""

    name = "launch_claude_code"
    description = (
        "Use only when the user asks to launch Claude Code on the Mac mini. "
        "This tool creates a pending request and returns the exact phrase the user "
        "must say next. It does not launch Claude Code directly."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["request", "cancel"],
                "description": "Use request for a new launch request, or cancel to clear a pending request.",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Create or cancel a pending launch request."""
        action = str(kwargs.get("action", "request")).strip().lower()
        controller = get_claude_code_launch_controller()
        if action == "request":
            return controller.request()
        if action == "cancel":
            return controller.cancel()
        return {"status": "error", "error": f"unsupported action: {action}"}
