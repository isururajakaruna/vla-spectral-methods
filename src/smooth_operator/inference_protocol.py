"""Length-prefixed local protocol for cross-environment VLA inference."""

import base64
import json
import socket
import struct
import zlib
from typing import Mapping

import numpy as np


MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("Inference connection closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection: socket.socket, message: dict) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Inference message exceeds size limit")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(connection: socket.socket) -> dict:
    size = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("Inference message exceeds size limit")
    return json.loads(_receive_exact(connection, size).decode("utf-8"))


def encode_request(image: np.ndarray, state: np.ndarray, prompt: str) -> dict:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Inference image must be RGB")
    return {
        "image_shape": list(image.shape),
        "image": base64.b64encode(zlib.compress(image.tobytes())).decode("ascii"),
        "state": np.asarray(state, dtype=float).tolist(),
        "prompt": prompt,
    }


def encode_multi_request(
    images: Mapping[str, np.ndarray], state: np.ndarray, prompt: str
) -> dict:
    if not images:
        raise ValueError("Inference request requires at least one camera")
    encoded = {}
    for key, value in images.items():
        image = np.asarray(value, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Inference camera {key!r} must be RGB")
        encoded[key] = {
            "shape": list(image.shape),
            "data": base64.b64encode(zlib.compress(image.tobytes())).decode("ascii"),
        }
    return {
        "images": encoded,
        "state": np.asarray(state, dtype=float).tolist(),
        "prompt": prompt,
    }


def decode_request(message: dict):
    if "images" in message:
        images = {}
        for key, encoded in message["images"].items():
            shape = tuple(int(value) for value in encoded["shape"])
            data = zlib.decompress(base64.b64decode(encoded["data"]))
            image = np.frombuffer(data, dtype=np.uint8).reshape(shape)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Invalid inference camera shape")
            images[str(key)] = image
        state = np.asarray(message["state"], dtype=np.float32)
        if state.ndim != 1 or state.size < 1:
            raise ValueError("Invalid inference state shape")
        return images, state, str(message["prompt"])
    shape = tuple(int(value) for value in message["image_shape"])
    data = zlib.decompress(base64.b64decode(message["image"]))
    image = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    state = np.asarray(message["state"], dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or state.shape != (4,):
        raise ValueError("Invalid inference request shape")
    return image, state, str(message["prompt"])


class SocketActionChunkPolicy:
    def __init__(self, host: str = "127.0.0.1", port: int = 5478, timeout: float = 30.0):
        self.connection = socket.create_connection((host, port), timeout=timeout)

    def predict_action_chunk(self, image, state, instruction) -> np.ndarray:
        request = (
            encode_multi_request(image, state, instruction)
            if isinstance(image, Mapping)
            else encode_request(image, state, instruction)
        )
        send_message(self.connection, request)
        response = receive_message(self.connection)
        if "error" in response:
            raise RuntimeError(response["error"])
        return np.asarray(response["actions"], dtype=float)

    def close(self) -> None:
        self.connection.close()
