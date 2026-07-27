"""Tests for BackgroundToolManager."""

from __future__ import annotations
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bobe.tools.tool_constants import ToolState
from bobe.tools.background_tool_manager import (
    BackgroundTool,
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_routine(
    tool_name: str = "test_tool",
    result: dict[str, Any] | None = None,
    error: Exception | None = None,
    delay: float = 0.0,
) -> ToolCallRoutine:
    """Create a mock ToolCallRoutine that returns *result* or raises *error*.

    If *delay* > 0, the routine will sleep for that many seconds before
    returning / raising so we can test cancellation and progress.

    Mirrors the contract of ``_dispatch_tool_call`` in core_tools: normal
    exceptions are caught and returned as ``{"error": "..."}`` dicts, while
    ``CancelledError`` re-raises so task cancellation propagates to the task.
    """
    routine = MagicMock(spec=ToolCallRoutine)
    routine.tool_name = tool_name
    routine.args_json_str = "{}"

    async def _call(manager: BackgroundToolManager) -> dict[str, Any]:
        try:
            if delay:
                await asyncio.sleep(delay)
            if error is not None:
                raise error
            return result or {"ok": True}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    routine.__call__ = _call  # type: ignore[method-assign]
    routine.side_effect = _call
    return routine


# ---------------------------------------------------------------------------
# Model / data-class sanity checks
# ---------------------------------------------------------------------------


class TestToolNotification:
    """Validate ToolNotification construction."""

    def test_creation(self) -> None:
        """Create a notification and verify its fields."""
        n = ToolNotification(
            id="abc",
            tool_name="my_tool",
            status=ToolState.COMPLETED,
            result={"data": 1},
        )
        assert n.id == "abc"
        assert n.status == ToolState.COMPLETED
        assert n.result == {"data": 1}
        assert n.error is None


class TestBackgroundTool:
    """Validate BackgroundTool helpers."""

    def test_tool_id(self) -> None:
        """Verify the composite tool_id property includes started_at."""
        t = BackgroundTool(
            id="123",
            tool_name="weather",
            status=ToolState.RUNNING,
        )
        assert t.tool_id == f"weather-123-{t.started_at}"

    def test_get_notification(self) -> None:
        """Convert a BackgroundTool to a ToolNotification."""
        t = BackgroundTool(
            id="1",
            tool_name="t",
            status=ToolState.COMPLETED,
            result={"x": 1},
            error=None,
        )
        n = t.get_notification()
        assert isinstance(n, ToolNotification)
        assert n.id == "1"
        assert n.tool_name == "t"
        assert n.status == ToolState.COMPLETED
        assert n.result == {"x": 1}


# ---------------------------------------------------------------------------
# BackgroundToolManager
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> BackgroundToolManager:
    """Return a fresh BackgroundToolManager for each test."""
    return BackgroundToolManager()


class TestStartTool:
    """Verify tool registration via start_tool."""

    @pytest.mark.asyncio
    async def test_start_registers_tool(self, manager: BackgroundToolManager) -> None:
        """Register a tool and verify its initial state."""
        routine = _make_routine("greet")
        bg = await manager.start_tool(
            call_id="c1",
            tool_call_routine=routine,
        )
        assert bg.tool_name == "greet"
        assert bg.id == "c1"
        assert bg.status == ToolState.RUNNING
        assert manager.get_tool(bg.tool_id) is bg

        # Let the task finish
        await asyncio.sleep(0.05)


class TestRunToolLifecycle:
    """Test _run_tool via start_tool (the public entry point)."""

    @pytest.mark.asyncio
    async def test_successful_completion(self, manager: BackgroundToolManager) -> None:
        """Complete a tool and verify result, status, and notification."""
        routine = _make_routine("ok_tool", result={"answer": 42})
        bg = await manager.start_tool("c1", routine)

        # Wait for the task to finish
        await asyncio.sleep(0.05)

        assert bg.status == ToolState.COMPLETED
        assert bg.result == {"answer": 42}
        assert bg.completed_at is not None
        assert bg.error is None

        # Notification should be queued
        notification = manager._notification_queue.get_nowait()
        assert notification.status == ToolState.COMPLETED

    @pytest.mark.asyncio
    async def test_tool_failure(self, manager: BackgroundToolManager) -> None:
        """Mark a tool as FAILED when it raises an exception."""
        routine = _make_routine("bad_tool", error=ValueError("boom"))
        bg = await manager.start_tool("c1", routine)

        await asyncio.sleep(0.05)

        assert bg.status == ToolState.FAILED
        assert "ValueError: boom" in (bg.error or "")
        assert bg.completed_at is not None

        notification = manager._notification_queue.get_nowait()
        assert notification.status == ToolState.FAILED

    @pytest.mark.asyncio
    async def test_tool_cancellation(self, manager: BackgroundToolManager) -> None:
        """Cancel a running tool and verify CANCELLED status."""
        routine = _make_routine("long_tool", delay=10.0)
        bg = await manager.start_tool("c1", routine)

        # Give the task a moment to start, then cancel
        await asyncio.sleep(0.02)
        cancelled = await manager.cancel_tool(bg.tool_id)
        assert cancelled is True

        # Let cancellation propagate
        await asyncio.sleep(0.05)

        assert bg.status == ToolState.CANCELLED
        assert bg.error == "Tool cancelled"
        assert bg.completed_at is not None


class TestCancelTool:
    """Verify tool cancellation behaviour."""

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, manager: BackgroundToolManager) -> None:
        """Return False when the tool_id does not exist."""
        result = await manager.cancel_tool("does-not-exist")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_already_completed(self, manager: BackgroundToolManager) -> None:
        """Return True when cancelling an already-completed tool."""
        routine = _make_routine("done")
        bg = await manager.start_tool("c1", routine)
        await asyncio.sleep(0.05)  # let it finish
        assert bg.status == ToolState.COMPLETED

        # Cancelling a completed tool should return True (not running, no-op)
        result = await manager.cancel_tool(bg.tool_id)
        assert result is True


class TestTimeoutTools:
    """Verify automatic timeout of long-running tools."""

    @pytest.mark.asyncio
    async def test_timeout_cancels_old_tools(self, manager: BackgroundToolManager) -> None:
        """Cancel tools exceeding max duration."""
        # Use a very short max duration
        manager._max_tool_duration_seconds = 0.01

        routine = _make_routine("slow", delay=10.0)
        await manager.start_tool("c1", routine)

        # Wait longer than the timeout
        await asyncio.sleep(0.05)

        count = await manager.timeout_tools()
        assert count == 1

        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_timeout_ignores_recent_tools(self, manager: BackgroundToolManager) -> None:
        """Leave recent tools untouched."""
        manager._max_tool_duration_seconds = 9999

        routine = _make_routine("fast", delay=10.0)
        bg = await manager.start_tool("c1", routine)

        count = await manager.timeout_tools()
        assert count == 0

        await manager.cancel_tool(bg.tool_id)
        await asyncio.sleep(0.05)


class TestCleanupTools:
    """Verify cleanup of completed tools from memory."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_completed(self, manager: BackgroundToolManager) -> None:
        """Remove completed tools past the retention window."""
        manager._max_tool_memory_seconds = 0.01

        routine = _make_routine("old")
        bg = await manager.start_tool("c1", routine)
        await asyncio.sleep(0.05)
        assert bg.status == ToolState.COMPLETED

        # Wait for the memory retention to expire
        await asyncio.sleep(0.05)

        removed = await manager.cleanup_tools()
        assert removed == 1
        assert manager.get_tool(bg.tool_id) is None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_recent_completed(self, manager: BackgroundToolManager) -> None:
        """Keep recently completed tools."""
        manager._max_tool_memory_seconds = 9999

        routine = _make_routine("recent")
        bg = await manager.start_tool("c1", routine)
        await asyncio.sleep(0.05)

        removed = await manager.cleanup_tools()
        assert removed == 0
        assert manager.get_tool(bg.tool_id) is not None

    @pytest.mark.asyncio
    async def test_cleanup_ignores_running(self, manager: BackgroundToolManager) -> None:
        """Never remove still-running tools."""
        manager._max_tool_memory_seconds = 0.0  # immediate expiry

        routine = _make_routine("still_going", delay=10.0)
        bg = await manager.start_tool("c1", routine)

        removed = await manager.cleanup_tools()
        assert removed == 0

        await manager.cancel_tool(bg.tool_id)
        await asyncio.sleep(0.05)


class TestGetters:
    """Verify tool retrieval helpers."""

    @pytest.mark.asyncio
    async def test_get_tool(self, manager: BackgroundToolManager) -> None:
        """Return None for missing tools and the instance for known ones."""
        assert manager.get_tool("nope") is None

        routine = _make_routine("x")
        bg = await manager.start_tool("1", routine)
        assert manager.get_tool(bg.tool_id) is bg
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_get_running_tools(self, manager: BackgroundToolManager) -> None:
        """Return only tools that are still running."""
        r1 = _make_routine("a", delay=10.0)
        r2 = _make_routine("b", delay=10.0)
        r3 = _make_routine("c")  # finishes immediately

        bg1 = await manager.start_tool("1", r1)
        bg2 = await manager.start_tool("2", r2)
        await manager.start_tool("3", r3)
        await asyncio.sleep(0.05)  # let r3 finish

        running = manager.get_running_tools()
        assert len(running) == 2
        names = {t.tool_name for t in running}
        assert names == {"a", "b"}

        # Clean up
        await manager.cancel_tool(bg1.tool_id)
        await manager.cancel_tool(bg2.tool_id)
        await asyncio.sleep(0.05)


class TestStartUp:
    """Verify start_up bootstraps background tasks."""

    @pytest.mark.asyncio
    async def test_startup_creates_tasks(self, manager: BackgroundToolManager) -> None:
        """start_up should create the listener and cleanup background tasks."""
        callback = AsyncMock()
        manager.start_up(tool_callbacks=[callback])

        # Start a tool and let it complete — the listener should invoke the callback
        routine = _make_routine("ping")
        await manager.start_tool("c1", routine)
        await asyncio.sleep(0.1)

        assert callback.call_count == 1
        notification = callback.call_args[0][0]
        assert isinstance(notification, ToolNotification)
        assert notification.status == ToolState.COMPLETED

    @pytest.mark.asyncio
    async def test_startup_multiple_callbacks(self, manager: BackgroundToolManager) -> None:
        """Invoke all registered callbacks on completion."""
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        manager.start_up(tool_callbacks=[cb1, cb2])

        routine = _make_routine("multi")
        await manager.start_tool("c1", routine)
        await asyncio.sleep(0.1)

        assert cb1.call_count == 1
        assert cb2.call_count == 1


class TestNotificationQueue:
    """Verify notifications are enqueued on tool completion or failure."""

    @pytest.mark.asyncio
    async def test_notifications_queued_on_completion(self, manager: BackgroundToolManager) -> None:
        """Queue a COMPLETED notification with the tool result."""
        routine = _make_routine("notif", result={"v": 1})
        await manager.start_tool("c1", routine)
        await asyncio.sleep(0.05)

        n = manager._notification_queue.get_nowait()
        assert n.tool_name == "notif"
        assert n.status == ToolState.COMPLETED
        assert n.result == {"v": 1}

    @pytest.mark.asyncio
    async def test_notifications_queued_on_failure(self, manager: BackgroundToolManager) -> None:
        """Queue a FAILED notification with the error message."""
        routine = _make_routine("fail", error=RuntimeError("oops"))
        await manager.start_tool("c1", routine)
        await asyncio.sleep(0.05)

        n = manager._notification_queue.get_nowait()
        assert n.status == ToolState.FAILED
        assert "RuntimeError: oops" in (n.error or "")

    @pytest.mark.asyncio
    async def test_legacy_cancelled_result_dict_still_marks_cancelled(self, manager: BackgroundToolManager) -> None:
        """Routines returning {"error": "Tool cancelled"} keep the CANCELLED status."""
        routine = MagicMock(spec=ToolCallRoutine)
        routine.tool_name = "legacy"
        routine.args_json_str = "{}"

        async def _call(_manager: BackgroundToolManager) -> dict[str, Any]:
            return {"error": "Tool cancelled"}

        routine.side_effect = _call

        bg = await manager.start_tool("c1", routine)
        await asyncio.sleep(0.05)

        assert bg.status == ToolState.CANCELLED
        n = manager._notification_queue.get_nowait()
        assert n.status == ToolState.CANCELLED


class TestListenerResilience:
    """Finding #4: the listener must survive callback exceptions."""

    @pytest.mark.asyncio
    async def test_listener_survives_callback_exception(self, manager: BackgroundToolManager) -> None:
        """A raising callback must not kill delivery for later notifications."""
        calls: list[ToolNotification] = []

        async def flaky(notification: ToolNotification) -> None:
            calls.append(notification)
            if len(calls) == 1:
                raise RuntimeError("boom")

        manager.start_up(tool_callbacks=[flaky])
        try:
            await manager.start_tool("c1", _make_routine("first"))
            await asyncio.sleep(0.05)
            await manager.start_tool("c2", _make_routine("second"))
            await asyncio.sleep(0.05)

            # The second notification is still delivered after the first raised.
            assert [n.tool_name for n in calls] == ["first", "second"]
            listener_tasks = [t for t in manager._lifecycle_tasks if "listener" in t.get_name()]
            assert listener_tasks and not listener_tasks[0].done()
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_swallows_listener_stored_exception(self, manager: BackgroundToolManager) -> None:
        """A lifecycle task that died with a stored exception must not re-raise out of shutdown."""
        manager.start_up(tool_callbacks=[AsyncMock()])

        async def _dead() -> None:
            raise RuntimeError("listener died (simulated)")

        dead_task = asyncio.get_running_loop().create_task(_dead(), name="bg-tool-listener-callback")
        await asyncio.sleep(0)
        assert dead_task.done()
        manager._lifecycle_tasks.append(dead_task)

        await manager.shutdown()  # must not raise

        assert manager._lifecycle_tasks == []


class TestShutdownOrdering:
    """Finding #6: tools stop before the listener; nothing replays into the next session."""

    @pytest.mark.asyncio
    async def test_shutdown_awaits_cancelled_tools_and_drains_notifications(
        self, manager: BackgroundToolManager
    ) -> None:
        """Cancelled-tool notifications never survive shutdown into the queue."""
        manager.start_up(tool_callbacks=[AsyncMock()])
        bg = await manager.start_tool("c1", _make_routine("slow", delay=10.0))
        await asyncio.sleep(0.02)

        await manager.shutdown()

        assert bg.status == ToolState.CANCELLED
        assert bg._task is not None and bg._task.done()
        assert manager._notification_queue.empty()
        assert manager._lifecycle_tasks == []

        # A fresh session's listener must never see the dead session's
        # cancelled-tool notification.
        replay_callback = AsyncMock()
        manager.start_up(tool_callbacks=[replay_callback])
        try:
            await asyncio.sleep(0.05)
            assert replay_callback.await_count == 0
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_start_up_drains_stranded_notifications(self, manager: BackgroundToolManager) -> None:
        """Notifications stranded without a listener are dropped by the next start_up."""
        await manager.start_tool("c1", _make_routine("orphan"))
        await asyncio.sleep(0.05)
        assert not manager._notification_queue.empty()

        callback = AsyncMock()
        manager.start_up(tool_callbacks=[callback])
        try:
            await asyncio.sleep(0.05)
            assert callback.await_count == 0
            assert manager._notification_queue.empty()
        finally:
            await manager.shutdown()


class TestGenerationScopedLifecycle:
    """A stale session's shutdown must never tear down a newer session's listener."""

    @pytest.mark.asyncio
    async def test_stale_generation_shutdown_is_a_noop(self, manager: BackgroundToolManager) -> None:
        """shutdown(generation=old) after a newer start_up leaves the new session intact."""
        old_generation = manager.start_up(tool_callbacks=[AsyncMock()])

        new_callback = AsyncMock()
        manager.start_up(tool_callbacks=[new_callback])
        new_tasks = list(manager._lifecycle_tasks)
        bg = await manager.start_tool("c1", _make_routine("live", delay=10.0))

        # The old session's teardown arrives late: it must not cancel the new
        # session's lifecycle tasks or its running tools.
        await manager.shutdown(generation=old_generation)

        assert all(not t.done() for t in new_tasks)
        assert bg.status == ToolState.RUNNING

        # The new listener still delivers results.
        await manager.start_tool("c2", _make_routine("quick"))
        await asyncio.sleep(0.05)
        assert new_callback.await_count == 1

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_start_up_cancels_previous_lifecycle_tasks(self, manager: BackgroundToolManager) -> None:
        """start_up is an idempotent takeover: old lifecycle tasks never linger."""
        manager.start_up(tool_callbacks=[AsyncMock()])
        old_tasks = list(manager._lifecycle_tasks)

        manager.start_up(tool_callbacks=[AsyncMock()])
        await asyncio.sleep(0.02)

        assert all(t.done() for t in old_tasks)
        assert all(not t.done() for t in manager._lifecycle_tasks)

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_matching_generation_shutdown_stops_lifecycle(self, manager: BackgroundToolManager) -> None:
        """shutdown(generation=current) still performs a full teardown."""
        generation = manager.start_up(tool_callbacks=[AsyncMock()])
        tasks = list(manager._lifecycle_tasks)

        await manager.shutdown(generation=generation)

        assert all(t.done() for t in tasks)
        assert manager._lifecycle_tasks == []


class TestDispatchCancellationContract:
    """core_tools dispatch must propagate CancelledError, not convert it to a result."""

    @pytest.mark.asyncio
    async def test_dispatch_tool_call_reraises_cancellation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cancelling a dispatched tool cancels the task instead of completing it."""
        from bobe.tools import core_tools

        # Initialization is explicit now; load the registry before patching it
        # so the lazy guard inside dispatch cannot rebuild (and wipe) it.
        core_tools.ensure_tools_loaded()

        started = asyncio.Event()

        async def _sleepy_tool(deps: Any, **kwargs: Any) -> dict[str, Any]:
            started.set()
            await asyncio.sleep(10.0)
            return {"ok": True}

        monkeypatch.setitem(core_tools.ALL_TOOLS, "sleepy_test_tool", _sleepy_tool)

        task = asyncio.create_task(
            core_tools.dispatch_tool_call(tool_name="sleepy_test_tool", args_json="{}", deps=MagicMock()),
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
