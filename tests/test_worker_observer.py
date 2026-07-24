import asyncio
from types import SimpleNamespace

import pytest

from pipecat.frames.frames import TextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.processors.frame_processor import FrameDirection


class _FailsOnceObserver(BaseObserver):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def on_push_frame(self, data: FramePushed):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("expected test failure")


def _frame_pushed(text: str):
    return FramePushed(
        source=SimpleNamespace(),
        destination=SimpleNamespace(),
        frame=TextFrame(text),
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


@pytest.mark.asyncio
async def test_proxy_acknowledges_failure_and_continues_processing():
    worker_observer = WorkerObserver()
    observer = _FailsOnceObserver()
    queue = asyncio.Queue()
    proxy_task = asyncio.create_task(worker_observer._proxy_task_handler(queue, observer))

    try:
        await queue.put(_frame_pushed("first"))
        await queue.put(_frame_pushed("second"))
        await asyncio.wait_for(queue.join(), timeout=0.5)

        assert observer.calls == 2
        assert not proxy_task.done()
    finally:
        proxy_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await proxy_task
