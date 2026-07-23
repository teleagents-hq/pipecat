#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for node-transition function-call deferral in OpenAI LLM services."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.llm import OpenAILLMService


def _make_service() -> OpenAILLMService:
    with patch.object(OpenAILLMService, "create_client"):
        return OpenAILLMService(api_key="test-key")


def _make_function_call(name: str, tool_call_id: str) -> FunctionCallFromLLM:
    return FunctionCallFromLLM(
        context=LLMContext(),
        tool_call_id=tool_call_id,
        function_name=name,
        arguments={},
    )


class _MockAsyncStream:
    def __init__(self, chunks: list[SimpleNamespace]):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self):
        pass


def _make_chunk(*, content: str | None = None, tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ],
    )


@pytest.mark.asyncio
async def test_non_transition_call_is_not_deferred_after_generated_text():
    service = _make_service()
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call("look_up_account", "call-ordinary")

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=True,
    )

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_node_transition_runs_immediately_after_whitespace_only_content():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    service.push_frame = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.get_chat_completions = AsyncMock(
        return_value=_MockAsyncStream(
            [
                _make_chunk(content="\n "),
                _make_chunk(
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="call-transition",
                            function=SimpleNamespace(
                                name="transition_to_next_node",
                                arguments="{}",
                            ),
                        )
                    ]
                ),
            ]
        )
    )

    await service._process_context(LLMContext())

    function_calls = service.run_function_calls.await_args.args[0]
    assert len(function_calls) == 1
    assert function_calls[0].function_name == "transition_to_next_node"
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_node_transition_call_is_deferred_after_generated_text():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=True,
    )

    service.run_function_calls.assert_not_awaited()
    assert service._pending_node_transition_function_calls == [function_call]


@pytest.mark.asyncio
async def test_node_transition_call_runs_immediately_without_generated_text():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )

    await service._run_or_defer_function_calls(
        [function_call],
        text_generated=False,
    )

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []


@pytest.mark.asyncio
async def test_mixed_batch_with_node_transition_is_deferred_together():
    service = _make_service()
    service.register_function(
        "transition_to_next_node",
        AsyncMock(),
        is_node_transition=True,
    )
    service.run_function_calls = AsyncMock()
    function_calls = [
        _make_function_call("look_up_account", "call-ordinary"),
        _make_function_call("transition_to_next_node", "call-transition"),
    ]

    await service._run_or_defer_function_calls(
        function_calls,
        text_generated=True,
    )

    service.run_function_calls.assert_not_awaited()
    assert service._pending_node_transition_function_calls == function_calls


@pytest.mark.asyncio
async def test_pending_node_transition_batch_runs_after_tts():
    service = _make_service()
    service.run_function_calls = AsyncMock()
    function_call = _make_function_call(
        "transition_to_next_node",
        "call-transition",
    )
    service._pending_node_transition_function_calls = [function_call]

    await service._run_pending_node_transition_function_calls()

    service.run_function_calls.assert_awaited_once_with([function_call])
    assert service._pending_node_transition_function_calls == []
