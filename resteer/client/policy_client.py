# Adapted from openpi (Apache-2.0), packages/openpi-client/src/openpi_client/websocket_client_policy.py @ 215abfb.
"""Client for the openpi websocket policy protocol.

Any server that speaks this protocol can be evaluated, e.g. openpi's ``scripts/serve_policy.py``:
the client connects, receives a msgpack-encoded metadata dict, then sends msgpack-encoded
observation dicts and receives action dicts whose ``"actions"`` entry is an
``[action_horizon, action_dim]`` array. A server that returns a text frame reports an error.

Differences from openpi's client: a connection timeout instead of waiting forever, and
no keepalive pings (see ``_connect``), and
``infer_batch``/``sample_actions`` for drawing many action samples of one observation (used by CMI).
Batched requests are only sent when the server advertises ``"infer_batch"`` in
``metadata["methods"]``; stock openpi servers do not, and get sequential ``infer`` calls instead.
"""

import logging
import time

import numpy as np
import websockets.sync.client

from resteer.client import msgpack_numpy

logger = logging.getLogger(__name__)


class PolicyServerError(RuntimeError):
    """The policy server is unreachable, disconnected, or reported an error."""


class PolicyClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = 8000,
        *,
        api_key: str | None = None,
        connect_timeout_s: float = 900.0,
    ) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._api_key = api_key
        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._connect(connect_timeout_s)

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def supports_batch(self) -> bool:
        return "infer_batch" in self._metadata.get("methods", [])

    def _connect(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        logger.info("Waiting for policy server at %s ...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                # No keepalive pings: servers run inference inside their event loop and cannot answer
                # pings while busy (e.g. the first call compiles the model for well over 20 s).
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers, ping_interval=None
                )
                return conn, msgpack_numpy.unpackb(conn.recv())
            except (ConnectionRefusedError, OSError) as e:
                if time.monotonic() > deadline:
                    raise PolicyServerError(f"No policy server at {self._uri} after {timeout_s:.0f}s") from e
                time.sleep(5)

    def _request(self, payload):
        try:
            self._ws.send(self._packer.pack(payload))
            response = self._ws.recv()
        except websockets.exceptions.WebSocketException as e:
            raise PolicyServerError(f"Lost connection to {self._uri}: {e}") from e
        if isinstance(response, str):
            raise PolicyServerError(f"Error in policy server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def infer(self, obs: dict) -> dict:
        return self._request(obs)

    def infer_batch(self, obs_batch: list[dict]) -> list[dict]:
        if not self.supports_batch:
            return [self.infer(obs) for obs in obs_batch]
        result = self._request(obs_batch)
        if not isinstance(result, list):
            raise PolicyServerError("Server returned a non-list response to a batch request")
        return result

    def sample_actions(self, obs: dict, num_samples: int, batch_size: int = 8) -> np.ndarray:
        """Draws ``num_samples`` action chunks for the same observation: [num_samples, horizon, action_dim]."""
        chunks = []
        for start in range(0, num_samples, batch_size):
            n = min(batch_size, num_samples - start)
            chunks.extend(out["actions"] for out in self.infer_batch([obs] * n))
        return np.stack(chunks, axis=0)

    def close(self) -> None:
        self._ws.close()
