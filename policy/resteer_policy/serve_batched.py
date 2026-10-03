"""Policy server with dynamic batching, for high-throughput evaluation and data collection.

Serves a trained openpi checkpoint over openpi's websocket protocol (same metadata frame, same
request/response format, ``GET /healthz``), like ``scripts/serve_policy.py policy:checkpoint``.
The difference: requests that arrive while the model is busy, e.g. from many simulation workers,
are run through the model together as one batch.

Input and output transforms are openpi's own and run per request exactly as in
``openpi.policies.policy.Policy.infer``; only the model call is batched. Batches are padded to a
power of two so that few shapes are compiled (all are compiled before the server starts
listening). Each request gets its own noise sample, drawn from one RNG split per batch, so the
outputs match sequential serving in distribution rather than bitwise.

The server also accepts a list of observations (batched request, used by CMI sampling) and
advertises this in its metadata (``"methods"``).

    scripts/policy.sh serve_batched --port 8000 --checkpoint gs://openpi-assets/checkpoints/pi05_libero
"""

import asyncio
import concurrent.futures
import dataclasses
import http
import logging
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import msgpack_numpy
import tyro
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    config: str = "pi05_libero"
    port: int = 8000
    max_batch: int = 32
    transform_threads: int = 8


class BatchedPolicy:
    """Runs openpi's transforms per request and the model once per batch."""

    def __init__(self, policy, max_batch: int):
        self._policy = policy
        self._rng = jax.random.key(0)
        self.sizes = [2**i for i in range(max_batch.bit_length()) if 2**i <= max_batch]

    def transform(self, obs: dict) -> dict:
        return self._policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001

    def run(self, inputs: list[dict]) -> list[dict]:
        n = len(inputs)
        size = next(s for s in self.sizes if s >= n)
        padded = inputs + [inputs[-1]] * (size - n)
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *padded)
        self._rng, rng = jax.random.split(self._rng)
        start = time.monotonic()
        actions = self._policy._sample_actions(  # noqa: SLF001
            rng,
            _model.Observation.from_dict(batch),
            **self._policy._sample_kwargs,  # noqa: SLF001
        )
        actions = np.asarray(actions)
        model_ms = 1000 * (time.monotonic() - start)
        state = np.asarray(batch["state"])
        outputs = []
        for i in range(n):
            out = self._policy._output_transform({"state": state[i], "actions": actions[i]})  # noqa: SLF001
            out["policy_timing"] = {"infer_ms": model_ms, "batch_size": n}
            outputs.append(out)
        return outputs

    def warmup(self, example_obs: dict) -> None:
        example = self.transform(example_obs)
        for size in self.sizes:
            start = time.monotonic()
            self.run([example] * size)
            logger.info("compiled batch size %d in %.1fs", size, time.monotonic() - start)


class BatchingServer:
    def __init__(self, policy: BatchedPolicy, metadata: dict, max_batch: int, transform_threads: int):
        self._policy = policy
        self._metadata = {**metadata, "methods": ["infer", "infer_batch"]}
        self._max_batch = max_batch
        self._transform_pool = concurrent.futures.ThreadPoolExecutor(transform_threads)
        self._model_pool = concurrent.futures.ThreadPoolExecutor(1)
        self._queue: asyncio.Queue | None = None

    async def _submit(self, obs: dict) -> dict:
        loop = asyncio.get_running_loop()
        inputs = await loop.run_in_executor(self._transform_pool, self._policy.transform, obs)
        future = loop.create_future()
        await self._queue.put((inputs, future))
        return await future

    async def _batcher(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = [await self._queue.get()]
            while len(items) < self._max_batch and not self._queue.empty():
                items.append(self._queue.get_nowait())
            try:
                outputs = await loop.run_in_executor(self._model_pool, self._policy.run, [x for x, _ in items])
                for (_, future), out in zip(items, outputs, strict=True):
                    future.set_result(out)
            except Exception as e:  # noqa: BLE001 - reported to every waiting request
                for _, future in items:
                    future.set_exception(e)

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                start = time.monotonic()
                if isinstance(request, list):
                    response = list(await asyncio.gather(*(self._submit(obs) for obs in request)))
                    items = response
                else:
                    response = await self._submit(request)
                    items = [response]
                for item in items:
                    item["server_timing"] = {"infer_ms": 1000 * (time.monotonic() - start)}
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Internal server error")
                raise

    async def serve(self, port: int) -> None:
        self._queue = asyncio.Queue()
        batcher = asyncio.create_task(self._batcher())

        def health_check(connection, request):
            return connection.respond(http.HTTPStatus.OK, "OK\n") if request.path == "/healthz" else None

        # No keepalive pings: clients are local simulation workers that can be busy (e.g. resetting a scene)
        # for longer than a ping timeout; a dead client is noticed when its socket closes.
        async with _server.serve(
            self._handler,
            "0.0.0.0",
            port,
            compression=None,
            max_size=None,
            process_request=health_check,
            ping_interval=None,
        ) as server:
            logger.info("Serving on port %d (max batch %d)", port, self._max_batch)
            await server.serve_forever()
        batcher.cancel()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)
    batched = BatchedPolicy(policy, args.max_batch)
    batched.warmup(
        {
            "observation/image": np.zeros((224, 224, 3), np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
            "observation/state": np.zeros(8),
            "prompt": "warm up",
        }
    )
    server = BatchingServer(batched, policy.metadata, args.max_batch, args.transform_threads)
    asyncio.run(server.serve(args.port))


if __name__ == "__main__":
    main(tyro.cli(Args))
