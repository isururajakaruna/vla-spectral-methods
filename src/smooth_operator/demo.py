"""Command-line entry point for the three-joint arm simulation."""

import argparse
import json
from pathlib import Path
import time

import pybullet as p

from .simulation import ManipulatorSimulation
from .motion import BasicPickPlaceController
from .inference_protocol import SocketActionChunkPolicy
from .vla import PromptResolver, SafeVLAExecutor


def _joint_sliders(simulation: ManipulatorSimulation) -> tuple[int, ...]:
    sliders = []
    for joint_index in simulation.joint_indices:
        info = p.getJointInfo(
            simulation.robot_id,
            joint_index,
            physicsClientId=simulation.client_id,
        )
        sliders.append(
            p.addUserDebugParameter(
                info[1].decode("utf-8"),
                info[8],
                info[9],
                0.0,
                physicsClientId=simulation.client_id,
            )
        )
    return tuple(sliders)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold", type=float, default=0.0)
    parser.add_argument("--no-sliders", action="store_true")
    parser.add_argument("--move", choices=("apple", "coke_can", "pen"))
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--controller", choices=("astar", "smolvla"), default="astar")
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--inference-host", default="127.0.0.1")
    parser.add_argument("--inference-port", type=int, default=5478)
    parser.add_argument("--result", type=Path, default=Path("artifacts/result.json"))
    parser.add_argument(
        "--chart", type=Path, default=Path("artifacts/basic_motion.png")
    )
    args = parser.parse_args()
    if args.move and args.prompt:
        parser.error("--move and --prompt cannot be used together")

    with ManipulatorSimulation(gui=not args.headless) as simulation:
        selected_object = args.move
        destination = None
        result = None
        if args.prompt:
            if args.controller == "smolvla":
                policy = SocketActionChunkPolicy(args.inference_host, args.inference_port)
                result = SafeVLAExecutor(
                    simulation, policy, args.checkpoint
                ).execute(args.prompt)
                policy.close()
                selected_object = result["object"]
            else:
                positions = {
                    name: p.getBasePositionAndOrientation(
                        body, physicsClientId=simulation.client_id
                    )[0][:2]
                    for name, body in simulation.object_ids.items()
                }
                action = PromptResolver().resolve(args.prompt, positions)
                selected_object = action.object_name
                destination = action.destination

        if selected_object and result is None:
            controller = BasicPickPlaceController(simulation)
            metrics = controller.move_object(
                selected_object,
                destination if destination is not None else (0.45, 0.35),
            )
            controller.save_results(args.chart, args.result)
            result = {
                "controller": "astar",
                "fallback_reason": None,
                "checkpoint": None,
                "prompt": args.prompt,
                "seed": None,
                "object": selected_object,
                "task_success": True,
                **metrics,
            }
        if result is not None:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))

        sliders = (
            _joint_sliders(simulation)
            if simulation.gui and not args.no_sliders and not selected_object
            else ()
        )
        end_time = time.monotonic() + args.hold if args.hold > 0 else None

        try:
            while p.isConnected(simulation.client_id):
                if sliders:
                    simulation.command_joints(
                        [
                            p.readUserDebugParameter(
                                slider, physicsClientId=simulation.client_id
                            )
                            for slider in sliders
                        ]
                    )
                simulation.step()
                if args.headless or selected_object or (
                    end_time is not None and time.monotonic() >= end_time
                ):
                    break
                time.sleep(simulation.TIME_STEP)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
