"""Background tool orchestrator for non-blocking tool execution.

Allows tools to run long operations asynchronously while the robot
continues conversing. Tools can be tracked, cancelled, and their
completion is announced vocally via a silent notification queue.
"""

from __future__ import annotations
import time
import asyncio
import logging
from typing import Any, Dict, Callable, Optional, Coroutine

from pydantic import Field, BaseModel, PrivateAttr

from bobe.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
    dispatch_tool_call_with_manager,
)
from bobe.tools.tool_constants import ToolState, SystemTool


logger = logging.getLogger(__name__)

_SYSTEM_TOOL_NAMES: set[str] = {t.value for t in SystemTool}

# How long shutdown() waits for cancelled tool tasks to actually finish.
_TOOL_SHUTDOWN_TIMEOUT_S: float = 5.0


def _consume_task_result(task: "asyncio.Task[Any]") -> None:
    """Retrieve a finished task's exception so it is never reported as unretrieved."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("Discarded exception from task %r: %r", task.get_name(), exc)


class ToolProgress(BaseModel):
    """Progress of a background tool."""

    progress: float = Field(..., ge=0.0, le=1.0)
    message: Optional[str] = None


class ToolCallRoutine(BaseModel):
    """Encapsulates an async callable with its arguments for deferred execution."""

    model_config = {"arbitrary_types_allowed": True}

    tool_name: str
    args_json_str: str
    deps: "ToolDependencies"

    async def __call__(self, tool_manager: BackgroundToolManager) -> Any:
        """Execute the stored callable with its arguments."""
        if self.tool_name in _SYSTEM_TOOL_NAMES:
            # For safety purposes, we only allow system tools to be called with the tool manager
            return await dispatch_tool_call_with_manager(tool_name=self.tool_name, args_json=self.args_json_str, deps=self.deps, tool_manager=tool_manager)
        return await dispatch_tool_call(tool_name=self.tool_name, args_json=self.args_json_str, deps=self.deps)


class ToolNotification(BaseModel):
    """Notification payload for completed tools."""

    id: str
    tool_name: str
    status: ToolState
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BackgroundTool(ToolNotification):
    """Represents a background tool."""

    progress: Optional[ToolProgress] = None
    started_at: float = Field(default_factory=time.monotonic)
    completed_at: Optional[float] = None

    # The async tool execution task.
    _task: Optional[asyncio.Task[None]] = PrivateAttr(default=None)

    @property
    def tool_id(self) -> str:
        """Get the name of the tool."""
        return f"{self.tool_name}-{self.id}-{self.started_at}"

    def get_notification(self) -> ToolNotification:
        """Get the notification for the tool."""
        return ToolNotification(
            id=self.id,
            tool_name=self.tool_name,
            status=self.status,
            result=self.result,
            error=self.error,
        )


class BackgroundToolManager(BaseModel):
    """Manages background tools for non-blocking tool execution.

    Features:
    - Start async tools without blocking the conversation
    - Track tool status and progress
    - Cancel running tools

    """

    _tools: Dict[str, BackgroundTool] = PrivateAttr(default_factory=dict)
    _notification_queue: asyncio.Queue[ToolNotification] = PrivateAttr(default_factory=asyncio.Queue)
    _loop: Optional[asyncio.AbstractEventLoop] = PrivateAttr(default=None)
    # Internal lifecycle tasks (notification listener, periodic cleanup).
    _lifecycle_tasks: list[asyncio.Task[None]] = PrivateAttr(default_factory=list)
    # Monotonic token identifying the start_up() that owns _lifecycle_tasks.
    # A stale session's shutdown(generation=...) must never touch tasks that
    # a newer start_up() created.
    _lifecycle_generation: int = PrivateAttr(default=0)
    # Tools running longer than this are auto-cancelled (1 day).
    _max_tool_duration_seconds: float = PrivateAttr(default=86400)
    # Completed/failed/cancelled tools older than this are purged (1 hour).
    _max_tool_memory_seconds: float = PrivateAttr(default=3600)

    def set_loop(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Set the event loop.

        Args:
            loop: The event loop (defaults to current running loop)

        """
        if loop is not None:
            self._loop = loop
        else:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        logger.debug("BackgroundToolManager: event loop set")


    async def start_tool(
        self,
        call_id: str,
        tool_call_routine: ToolCallRoutine,
        with_progress: bool = False,
    ) -> BackgroundTool:
        """Start a new background tool.

        Args:
            call_id: The ID of the tool
            tool_call_routine: The ToolCallRoutine containing the callable and its arguments
            with_progress: Whether to track progress (0.0-1.0)

        Returns:
            BackgroundTool object with tool ID

        """
        tool_name = tool_call_routine.tool_name
        id = call_id
        bg_tool = BackgroundTool(
            id=id,
            tool_name=tool_name,
            progress=ToolProgress(progress=0.0) if with_progress else None,
            status=ToolState.RUNNING,
        )
        self._tools[bg_tool.tool_id] = bg_tool

        async_task = asyncio.create_task(
            self._run_tool(bg_tool, tool_call_routine),
            name=f"bg-{tool_name}-{id}",
        )
        bg_tool._task = async_task

        logger.info(f"Started background tool: {bg_tool.tool_name} (id={id})")

        return bg_tool

    async def _run_tool(
        self,
        bg_tool: BackgroundTool,
        tool_call_routine: ToolCallRoutine,
    ) -> None:
        """Execute the tool and handle completion."""
        try:
            result: dict[str, Any] = await tool_call_routine(self)
        except asyncio.CancelledError:
            # Cancellation propagates from the tool (core_tools re-raises it).
            # Record the outcome, queue the notification synchronously (the
            # queue is unbounded), and let the cancellation finish the task.
            bg_tool.completed_at = time.monotonic()
            bg_tool.status = ToolState.CANCELLED
            bg_tool.error = "Tool cancelled"
            self._notification_queue.put_nowait(bg_tool.get_notification())
            logger.debug(f"Background tool cancelled: {bg_tool.tool_name} (id={bg_tool.id})")
            raise
        bg_tool.completed_at = time.monotonic()
        error = result.get("error")

        if error is not None:
            if error == "Tool cancelled":
                bg_tool.status = ToolState.CANCELLED
                logger.debug(f"Background tool cancelled: {bg_tool.tool_name} (id={bg_tool.id})")
            else:
                bg_tool.status = ToolState.FAILED
                logger.debug(f"Background tool failed: {bg_tool.tool_name} (id={bg_tool.id}): {bg_tool.error}")
            bg_tool.error = result["error"]

        else:
            bg_tool.result = result
            bg_tool.status = ToolState.COMPLETED
            logger.debug(f"Background tool completed: {bg_tool.tool_name} (id={bg_tool.id})")

        await self._notification_queue.put(bg_tool.get_notification())
        logger.debug(f"Queued notification for tool: {bg_tool.tool_name} (id={bg_tool.id})")

    async def update_progress(
        self,
        tool_id: str,
        progress: float,
        message: Optional[str] = None,
    ) -> bool:
        """Update progress for a tool (for tools with with_progress=True).

        Args:
            tool_id: The tool ID
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message (e.g., "50% downloaded")

        Returns:
            True if updated successfully, False if tool not found or not tracking progress

        """
        tool = self._tools.get(tool_id)
        if tool is None:
            return False

        if tool.progress is None:
            # Tool not tracking progress
            return False

        tool.progress = ToolProgress(progress=max(0.0, min(1.0, progress)), message=message)
        logger.debug(f"Tool {tool_id} progress: {progress:.1%} - {message or ''}")
        return True

    async def cancel_tool(self, tool_id: str, log: bool = True) -> bool:
        """Cancel a running tool by ID.

        Args:
            tool_id: The tool ID to cancel
            log: Whether to log the cancellation

        Returns:
            True if cancelled, False if tool not found or not running

        """
        tool = self._tools.get(tool_id)
        if tool is None:
            if log:
                logger.warning(f"Cannot cancel tool {tool_id}: not found")
            return False

        if tool.status != ToolState.RUNNING:
            if log:
                logger.warning(f"Cannot cancel tool {tool_id}: status is {tool.status.value}")
            return True

        if tool._task:
            tool._task.cancel()
            if log:
                logger.info(f"Cancelled tool: {tool.tool_name} (id={tool_id})")
            return True

        return False

    def start_up(self, tool_callbacks: list[Callable[[ToolNotification], Coroutine[Any, Any, None]]]) -> int:
        """Start the background tool manager.

        This method starts two concurrent tasks:
        - _listener: Listens for completed BackgroundTool notifications and calls the callbacks.
        - _cleanup: Cleans up completed/failed/cancelled tools that have been in memory for too long and times out tools that have been running too long.

        Args:
            tool_callbacks: A list of async or sync callables that receive the completed BackgroundTool notifications.

        Returns:
            A generation token identifying this start_up. Pass it to
            ``shutdown(generation=...)`` so a stale caller can never tear down
            lifecycle tasks created by a newer ``start_up``.

        """
        self.set_loop()

        # Idempotent takeover: a previous session's lifecycle tasks must never
        # survive a new start_up (and its later shutdown must not touch ours).
        for stale_task in self._lifecycle_tasks:
            if not stale_task.done():
                stale_task.cancel()
            stale_task.add_done_callback(_consume_task_result)
        self._lifecycle_tasks = []
        self._lifecycle_generation += 1
        generation = self._lifecycle_generation

        # Notifications queued for a dead session's conversation must never
        # replay into this fresh session.
        self._drain_notifications()

        async def _listener() -> None:
            while True:
                bg_tool = await self._notification_queue.get()
                for callback in tool_callbacks:
                    try:
                        await callback(bg_tool)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # One bad notification/callback must not kill delivery
                        # for the rest of the session.
                        logger.exception(
                            "Tool notification callback failed for %s (id=%s)",
                            bg_tool.tool_name,
                            bg_tool.id,
                        )

        async def _cleanup(interval_seconds: float = 5 * 60) -> None:
            while True:
                await asyncio.sleep(interval_seconds)
                await self.cleanup_tools()
                await self.timeout_tools()

        self._lifecycle_tasks = [
            asyncio.create_task(_cleanup(), name="bg-tool-cleanup"),
            asyncio.create_task(_listener(), name="bg-tool-listener-callback"),
        ]

        logger.info(
            "BackgroundToolManager started. "
            "Max tool execution duration: %s seconds (tools running longer will be auto-cancelled). "
            "Max tool memory retention: %s seconds (completed/failed/cancelled tools older than this are purged).",
            self._max_tool_duration_seconds, self._max_tool_memory_seconds,
        )
        return generation

    def _drain_notifications(self) -> int:
        """Discard queued notifications; they belong to an ended session."""
        drained = 0
        while True:
            try:
                self._notification_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained += 1
        if drained:
            logger.debug("Discarded %d stale tool notification(s)", drained)
        return drained

    async def shutdown(self, generation: int | None = None) -> None:
        """Cancel running tools, then the background tasks (listener, cleanup).

        Ordering matters: tools are cancelled and awaited *before* the listener
        so their cancellation notifications are consumed (or drained below)
        instead of surviving in the persistent queue and replaying into the
        next session's fresh listener.

        Args:
            generation: When provided, only shut down if it still matches the
                generation returned by the owning ``start_up``. A stale caller
                (e.g. an old session's teardown racing a new session) becomes a
                no-op instead of killing the new session's listener and tools.

        """
        if generation is not None and generation != self._lifecycle_generation:
            logger.debug(
                "Skipping shutdown for stale generation %d (current %d)",
                generation,
                self._lifecycle_generation,
            )
            return

        # Cancel and await running tools first so every cancellation
        # notification is enqueued before the queue is drained.
        tool_tasks = [
            tool._task
            for tool in self._tools.values()
            if tool.status == ToolState.RUNNING and tool._task is not None and not tool._task.done()
        ]
        for tool_id in list(self._tools):
            await self.cancel_tool(tool_id, log=False)
        if tool_tasks:
            done, pending = await asyncio.wait(tool_tasks, timeout=_TOOL_SHUTDOWN_TIMEOUT_S)
            for task in done:
                _consume_task_result(task)
            for task in pending:
                logger.warning(
                    "Background tool task %r did not stop within %.1fs",
                    task.get_name(),
                    _TOOL_SHUTDOWN_TIMEOUT_S,
                )

        lifecycle_tasks = self._lifecycle_tasks
        self._lifecycle_tasks = []
        for task in lifecycle_tasks:
            task.cancel()
        for task in lifecycle_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # A listener that died with a stored exception (e.g. a closed
                # websocket) must not let that exception escape shutdown.
                logger.exception("Lifecycle task %r failed during shutdown", task.get_name())

        # Anything still queued belongs to this ended session; never let it
        # replay into the next one.
        self._drain_notifications()

        logger.info("BackgroundToolManager shut down")

    async def timeout_tools(self) -> int:
        """Cancel tools that have been running too long.

        Returns:
            Number of tools cancelled

        """
        now = time.monotonic()
        to_cancel = []

        for tool_id, tool in self._tools.items():
            if tool.status == ToolState.RUNNING:
                if tool.started_at and (now - tool.started_at) > self._max_tool_duration_seconds:
                    to_cancel.append(tool_id)

        for tool_id in to_cancel:
            await self.cancel_tool(tool_id)

        if to_cancel:
            logger.debug(f"Timed out {len(to_cancel)} tools")

        return len(to_cancel)

    async def cleanup_tools(self) -> int:
        """Remove completed/failed/cancelled tools that have been in memory for too long.

        Returns:
            Number of tools removed

        """
        now = time.monotonic()
        to_remove = []

        for tool_id, tool in self._tools.items():
            if tool.status in (ToolState.COMPLETED, ToolState.FAILED, ToolState.CANCELLED):
                if tool.completed_at and (now - tool.completed_at) > self._max_tool_memory_seconds:
                    to_remove.append(tool_id)

        for tool_id in to_remove:
            del self._tools[tool_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tools")

        return len(to_remove)

    def get_tool(self, tool_id: str) -> Optional[BackgroundTool]:
        """Get a tool by ID."""
        return self._tools.get(tool_id)

    def get_running_tools(self) -> list[BackgroundTool]:
        """Get all currently running tools."""
        return [t for t in self._tools.values() if t.status == ToolState.RUNNING]

    def get_all_tools(self, limit: Optional[int] = None) -> list[BackgroundTool]:
        """Get recent tools (most recent first).

        Args:
            limit: Maximum number of tools to return (None means all)

        Returns:
            List of tools sorted by start time (most recent first)

        """
        sorted_tools = sorted(
            self._tools.values(),
            key=lambda t: t.started_at,
            reverse=True,
        )
        if limit is not None:
            return sorted_tools[:limit]
        return sorted_tools
