"""Serve VLA action chunks to the PyBullet simulator over a local socket."""

import argparse
import json
import socket
import sys
import traceback

from .inference_protocol import decode_request, receive_message, send_message
from .lerobot_policy import LeRobotVLAPolicy, gated_repo_message
from .vla_models import resolve_spec


def serve(spec: dict, host: str, port: int, device: str) -> None:
    try:
        policy = LeRobotVLAPolicy(spec, device=device)
    except Exception as error:
        message = gated_repo_message(error)
        if message is None:
            raise
        print(f"FATAL: {message}", flush=True)
        sys.exit(1)
    rollout = None
    if policy.embodiment == "libero":
        from .libero_env import LiberoRollout

        rollout = LiberoRollout(policy)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        print(f"{spec['name']} inference server listening on {host}:{port}", flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        request = receive_message(connection)
                    except ConnectionError:
                        break
                    try:
                        if request.get("type") == "libero_rollout":
                            if rollout is None:
                                raise ValueError(f"{spec['name']} is not a LIBERO checkpoint")
                            send_message(connection, rollout.run(request))
                            continue
                        images, state, prompt = decode_request(request)
                        if not isinstance(images, dict):
                            raise ValueError("VLA inference needs the multi-camera request format")
                        actions, spectrum = policy.predict(images, state, prompt)
                        response = {"actions": actions.tolist()}
                        if spectrum is not None:
                            response["spectrum"] = spectrum
                        send_message(connection, response)
                    except Exception as error:
                        traceback.print_exc()
                        send_message(connection, {"error": f"{type(error).__name__}: {error}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="smolvla_base", help="Entry in smooth_operator.vla_models")
    parser.add_argument("--checkpoint", help="Override the checkpoint (e.g. a local fine-tune)")
    parser.add_argument("--spec-json", help="Full model spec as JSON; overrides --model-id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5478)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.spec_json:
        spec = json.loads(args.spec_json)
    else:
        spec = resolve_spec(args.model_id, args.checkpoint)
        if args.checkpoint:
            spec["stats"] = "checkpoint"
    serve(spec, args.host, args.port, args.device)


if __name__ == "__main__":
    main()
