# ruff: noqa: D103

from types import SimpleNamespace

import pytest

from bobe.tools.stop_dance import StopDance
from bobe.tools.stop_emotion import StopEmotion


class FakeMovementManager:
    def __init__(self):
        self.clear_count = 0

    def clear_move_queue(self):
        self.clear_count += 1


@pytest.mark.parametrize("tool_cls", [StopDance, StopEmotion])
def test_stop_tools_do_not_require_dummy_arguments(tool_cls):
    schema = tool_cls.parameters_schema

    assert schema["properties"] == {}
    assert schema["required"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_cls", [StopDance, StopEmotion])
async def test_stop_tools_clear_move_queue_without_arguments(tool_cls):
    movement_manager = FakeMovementManager()
    deps = SimpleNamespace(movement_manager=movement_manager)

    result = await tool_cls()(deps)

    assert movement_manager.clear_count == 1
    assert result["status"]
