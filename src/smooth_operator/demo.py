"""Run the initial scripted manipulation demo."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .simulation import PlanarArm, SimulationClosed


def run_demo():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without a desktop window")
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--hold", type=float, default=10, help="Seconds to keep the GUI open after completion")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with PlanarArm(gui=not args.headless) as sim:
        Image.fromarray(sim.camera()).save(args.output / "01_initial.png")
        sim.move_to(sim.start)
        sim.attach()
        Image.fromarray(sim.camera()).save(args.output / "02_attached.png")
        sim.move_to(sim.goal, seconds=3)
        sim.release()
        sim.move_to((0.60, 0.0), seconds=1)
        for _ in range(240):
            sim.step()
        error = float(np.linalg.norm(sim.object_position()[:2] - sim.goal))
        Image.fromarray(sim.camera()).save(args.output / "03_placed.png")
        result = {"controller": "scripted_quintic_joint_interpolation", "success": error < 0.03,
                  "placement_error_m": error, "simulation_time_s": sim.steps * sim.dt,
                  "object_position_m": sim.object_position().tolist(), "goal_xy_m": list(sim.goal)}
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if sim.gui:
            for _ in range(max(0, round(args.hold / sim.dt))):
                sim.step()
        if not result["success"]:
            raise SystemExit("Placement failed")


def main():
    try:
        run_demo()
    except (SimulationClosed, KeyboardInterrupt):
        print("Simulation closed.")


if __name__ == "__main__":
    main()
