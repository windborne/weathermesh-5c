from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Generic, TypeVar

import torch

from .output import DecodedProducts

Metadata = TypeVar("Metadata")
TensorKey = tuple[tuple[int, ...], torch.dtype]


@dataclass(frozen=True)
class _TransferTask(Generic[Metadata]):
    metadata: Metadata
    products: DecodedProducts
    event: torch.cuda.Event | None
    buffers: tuple[tuple[TensorKey, torch.Tensor], ...]


class _PinnedPool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._available: dict[TensorKey, queue.Queue[torch.Tensor]] = {}
        self._counts: dict[TensorKey, int] = {}

    def acquire(self, source: torch.Tensor) -> tuple[TensorKey, torch.Tensor]:
        key = (tuple(source.shape), source.dtype)
        available = self._available.setdefault(key, queue.Queue(self.capacity))
        try:
            return key, available.get_nowait()
        except queue.Empty:
            count = self._counts.get(key, 0)
            if count < self.capacity:
                self._counts[key] = count + 1
                return key, torch.empty(
                    source.shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=source.is_cuda,
                )
            return key, available.get()

    def release(self, key: TensorKey, buffer: torch.Tensor) -> None:
        self._available[key].put(buffer)


class AsyncProductConsumer(Generic[Metadata]):
    """Bounded asynchronous GPU transfer, matching the realtime producer pattern."""

    def __init__(
        self,
        consume: Callable[[Metadata, DecodedProducts], None],
        *,
        capacity: int = 3,
    ):
        if capacity < 1:
            raise ValueError("Transfer capacity must be positive")
        self._consume = consume
        self._pool = _PinnedPool(capacity)
        self._tasks: queue.Queue[_TransferTask[Metadata] | None] = queue.Queue(capacity)
        self._copy_streams: dict[torch.device, torch.cuda.Stream] = {}
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._work,
            name="wm5c-product-consumer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, metadata: Metadata, products: DecodedProducts) -> None:
        self._raise_if_failed()
        copied = {}
        buffers = []
        sources = []
        try:
            for field in fields(products):
                source = getattr(products, field.name)
                if source is None:
                    copied[field.name] = None
                    continue
                source = source.detach()
                key, buffer = self._pool.acquire(source)
                copied[field.name] = buffer
                buffers.append((key, buffer))
                sources.append((source, buffer))

            event = None
            cuda_sources = [pair for pair in sources if pair[0].is_cuda]
            if cuda_sources:
                devices = {source.device for source, _ in cuda_sources}
                if len(devices) != 1 or len(cuda_sources) != len(sources):
                    raise ValueError("Decoded products must use one common device")
                device = devices.pop()
                if device not in self._copy_streams:
                    self._copy_streams[device] = torch.cuda.Stream(device=device)
                copy_stream = self._copy_streams[device]
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_event(ready)
                    for source, buffer in sources:
                        buffer.copy_(source, non_blocking=True)
                        source.record_stream(copy_stream)
                    event = torch.cuda.Event()
                    event.record(copy_stream)
            else:
                for source, buffer in sources:
                    buffer.copy_(source)

            task = _TransferTask(
                metadata,
                DecodedProducts(**copied),
                event,
                tuple(buffers),
            )
            self._tasks.put(task)
        except BaseException:
            for key, buffer in buffers:
                self._pool.release(key, buffer)
            raise
        self._raise_if_failed()

    def _work(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                if task is None:
                    return
                if task.event is not None:
                    task.event.synchronize()
                if self._error is None:
                    self._consume(task.metadata, task.products)
            except BaseException as error:
                if self._error is None:
                    self._error = error
            finally:
                if task is not None:
                    for key, buffer in task.buffers:
                        self._pool.release(key, buffer)
                self._tasks.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Asynchronous product aggregation failed") from self._error

    def flush(self) -> None:
        self._tasks.join()
        self._raise_if_failed()

    def close(self) -> None:
        error = None
        try:
            self.flush()
        except BaseException as caught:
            error = caught
        self._tasks.put(None)
        self._tasks.join()
        self._thread.join()
        if error is not None:
            raise error

    def __enter__(self) -> AsyncProductConsumer[Metadata]:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
            return
        self._tasks.join()
        self._tasks.put(None)
        self._tasks.join()
        self._thread.join()
