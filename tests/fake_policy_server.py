"""Deterministic stand-in for an openpi policy server (unit tests and the launcher check).

Actions are a pure function of a digest of the request (images, state, prompt), so two clients
that behave identically send identical request sequences and get identical rollouts. Every
request digest is appended to ``--log`` for comparison. Lists are answered element-wise (batched
inference) when ``--batch`` is set. Like openpi's server, it answers ``GET /healthz``.

    python tests/fake_policy_server.py --port 8765 --log /tmp/requests.txt
"""

import argparse
import asyncio
import hashlib
import http

import numpy as np
import websockets.asyncio.server

from resteer.client import msgpack_numpy

ACTION_HORIZON = 10


def digest(obs: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(obs):
        value = obs[key]
        h.update(key.encode())
        h.update(np.asarray(value).tobytes() if not isinstance(value, str) else value.encode())
    return h.hexdigest()


def fake_actions(obs: dict) -> np.ndarray:
    rng = np.random.default_rng(int(digest(obs)[:16], 16))
    actions = rng.uniform(-0.3, 0.3, size=(ACTION_HORIZON, 7)).astype(np.float32)
    actions[:, 6] = np.where(rng.random(ACTION_HORIZON) > 0.5, 1.0, -1.0)
    return actions


async def serve(port: int, log_path: str | None, batch: bool) -> None:
    packer = msgpack_numpy.Packer()
    metadata = {"fake": True, **({"methods": ["infer", "infer_batch"]} if batch else {})}

    async def handler(ws):
        await ws.send(packer.pack(metadata))
        async for message in ws:
            request = msgpack_numpy.unpackb(message)
            items = request if isinstance(request, list) else [request]
            if log_path:
                with open(log_path, "a") as f:
                    f.writelines(f"{digest(obs)} {obs.get('prompt', '')}\n" for obs in items)
            outs = [{"actions": fake_actions(obs)} for obs in items]
            await ws.send(packer.pack(outs if isinstance(request, list) else outs[0]))

    def health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async with websockets.asyncio.server.serve(
        handler, "127.0.0.1", port, compression=None, max_size=None, process_request=health_check
    ):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log", default=None)
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    asyncio.run(serve(args.port, args.log, args.batch))
