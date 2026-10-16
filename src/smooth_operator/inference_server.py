"""Serve SmolVLA action chunks to the Python 3.9 simulator."""

import argparse
import socket

from .inference_protocol import decode_request, receive_message, send_message
from .smolvla_policy import SmolVLAPolicy


def serve(checkpoint: str, host: str, port: int, device: str) -> None:
    policy = SmolVLAPolicy(checkpoint, device=device)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        print(f"SmolVLA inference server listening on {host}:{port}", flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        request = receive_message(connection)
                    except ConnectionError:
                        break
                    try:
                        image, state, prompt = decode_request(request)
                        actions = policy.predict_action_chunk(image, state, prompt)
                        send_message(connection, {"actions": actions.tolist()})
                    except Exception as error:
                        send_message(connection, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5478)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    serve(args.checkpoint, args.host, args.port, args.device)


if __name__ == "__main__":
    main()
