"""Serial sync MCP client backed by one lifetime-owning async task.

The owner alone enters, uses and exits all AnyIO/MCP contexts. Deadlines cancel
the outer connection lifetime; after unwinding, the next bounded attempt starts
a new subprocess. Never enter a task group in one task and exit it in another.
"""
import asyncio
import concurrent.futures
import math
import threading
import time
from contextlib import AsyncExitStack, contextmanager
from datetime import timedelta

import anyio
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from src.evidence_mcp_client import MCPContractError, MCPCorpusTools
from src.evidence_telemetry import span


@contextmanager
def request_context(context):
    if context is None:
        yield
    else:
        from opentelemetry.context import attach, detach
        token = attach(context)
        try:
            yield
        finally:
            detach(token)


class PersistentMCPCorpusTools(MCPCorpusTools):
    """Reuse one server/index per explicitly closed serial client (usually a batch)."""
    CLEANUP_GRACE = 8.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding += ';lifecycle=reused-v1'
        self._thread = None
        self._ready = threading.Event()
        self._serial = threading.Lock()
        self._closed = False
        self._fatal = None
        self._stop_requested = False
        self._shutdown_complete = False
        self._active_scope = None
        self._loop = None
        self._queue = None
        self.connections_started = 0

    def _call(self, name, arguments):
        # Reject shared concurrent use rather than allowing unbounded queue delay.
        if not self._serial.acquire(blocking=False):
            raise RuntimeError('MCP client supports one serial caller')
        try:
            if self._closed:
                raise RuntimeError('MCP client is closed')
            return super()._call(name, arguments)
        finally:
            self._serial.release()

    def _invoke(self, name, arguments):
        if self._closed:
            raise OSError('MCP client closed after uncertain transport outcome')
        if self._thread is None:
            self._thread = threading.Thread(target=self._thread_main,
                name='evidence-mcp-owner', daemon=True)
            self._thread.start()
            if not self._ready.wait(self.timeout):
                self._closed = True
                raise OSError('MCP owner did not start before deadline')
        if self._fatal or not self._thread.is_alive():
            raise OSError('MCP owner stopped') from self._fatal
        future = concurrent.futures.Future()
        context = None
        if self.telemetry:
            from opentelemetry.context import get_current
            context = get_current()
        request = (name, arguments, future, context, time.monotonic() + self.timeout)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, request)
        try:
            return future.result(timeout=self.timeout + self.CLEANUP_GRACE)
        except concurrent.futures.TimeoutError as exc:
            if future.done():
                return future.result()  # owner-reported request timeout, not bridge timeout
            # A second synchronous fence covers unexpected owner/cleanup stalls.
            # Fail closed: no new requests while outcome/cleanup is uncertain.
            self._closed = True
            self._loop.call_soon_threadsafe(self._cancel_and_stop)
            raise OSError('MCP owner deadline/cleanup exceeded; client closed') from exc

    def _cancel_and_stop(self):
        self._stop_requested = True
        if self._active_scope:
            self._active_scope.cancel()
        self._enqueue_stop()

    def _enqueue_stop(self):
        # At most one serial request plus a shutdown sentinel can be queued.
        if not self._queue.full():
            self._queue.put_nowait(None)

    def _thread_main(self):
        try:
            asyncio.run(self._owner())
        except BaseException as exc:
            self._fatal = exc
        finally:
            self._ready.set()

    async def _owner(self):
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=2)
        self._ready.set()
        if self._stop_requested:
            return
        while True:
            request = await self._queue.get()
            if request is None:
                return
            failure = None
            self._pending_request = request
            try:
                # This outer scope outlives every entered SDK task group. Updating
                # its deadline preserves LIFO cancellation ownership across calls.
                with anyio.CancelScope(deadline=request[4]) as lifetime:
                    self._active_scope = lifetime
                    async with AsyncExitStack() as stack:
                        with request_context(request[3]), span(self.telemetry, 'mcp.connect'):
                            reader, writer = await stack.enter_async_context(stdio_client(self.parameters))
                            session = await stack.enter_async_context(ClientSession(reader, writer,
                                read_timeout_seconds=timedelta(seconds=self.timeout)))
                            self.connections_started += 1
                            await session.initialize()
                            listed = await session.list_tools()
                            if {t.name for t in listed.tools} != {'search_corpus', 'get_evidence'}:
                                raise MCPContractError('tool capability mismatch')
                        await self._serve_requests(session, request, lifetime)
                        return
                if lifetime.cancel_called:
                    failure = TimeoutError('MCP attempt deadline exceeded')
            except Exception as exc:
                failure = exc
            finally:
                self._active_scope = None
            # _serve_requests updates the current request for failure attribution.
            request = self._pending_request
            self._pending_request = None
            if request is not None and not request[2].done():
                request[2].set_exception(failure or OSError('MCP session closed'))
            elif self._stop_requested and failure:
                raise failure

    async def _serve_requests(self, session, request, lifetime):
        while request is not None:
            self._pending_request = request
            name, arguments, future, context, deadline = request
            lifetime.deadline = deadline
            with request_context(context):
                response = await session.call_tool(name, arguments)
                value = self.validate(response, name, arguments)
            future.set_result(value)
            self._pending_request = None
            lifetime.deadline = math.inf
            request = await self._queue.get()
        self._pending_request = None

    def close(self):
        if self._shutdown_complete:
            return
        if not self._serial.acquire(blocking=False):
            raise RuntimeError('cannot close MCP client during a call')
        try:
            self._closed = True
            self._stop_requested = True
            if self._thread is None:
                return
            with span(self.telemetry, 'mcp.close'):
                if self._thread.is_alive():
                    if not self._ready.wait(self.CLEANUP_GRACE):
                        raise OSError('MCP owner startup/cleanup incomplete')
                    if self._loop is not None and not self._loop.is_closed():
                        self._loop.call_soon_threadsafe(self._enqueue_stop)
                    self._thread.join(timeout=self.timeout + self.CLEANUP_GRACE)
                if self._thread.is_alive():
                    self._loop.call_soon_threadsafe(self._cancel_and_stop)
                    self._thread.join(timeout=self.CLEANUP_GRACE)
                if self._thread.is_alive():
                    raise OSError('MCP owner cleanup incomplete')
                if self._fatal:
                    raise OSError('MCP owner exited unexpectedly') from self._fatal
                self._shutdown_complete = True
        finally:
            self._serial.release()
