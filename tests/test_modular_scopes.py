import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from smooth_operator.dataset import filter_episode_actions, validate_raw_episode
from smooth_operator.eval_matrix import run_evaluation_matrix
from smooth_operator.metrics import (
    compute_chunk_boundary_jump,
    compute_comprehensive_metrics,
    compute_ldlj,
    compute_sparc,
    compute_spectral_features,
)
from smooth_operator.pipeline import (
    ModularExecutionPipeline,
    PrecomputedChunkPolicy,
    SceneSpec,
    apply_scene_spec,
    generate_oracle_action_sequence,
    resolve_scene_from_prompt,
)
from smooth_operator.raw_dataset import generate_raw_episodes
from smooth_operator.so101 import SO101Simulation
from smooth_operator.spectrum import (
    create_spectrum_module,
    dct_ii,
    idct_iii,
    parse_freq_band,
)
from smooth_operator.training import (
    fit_spectral_dataset_stats,
    resolve_training_spectrum_schema,
)


class SpectrumModuleTests(unittest.TestCase):
    def test_orthonormal_dct_round_trip(self):
        rng = np.random.default_rng(7)
        chunk = rng.normal(size=(10, 6))
        reconstructed = idct_iii(dct_ii(chunk))
        np.testing.assert_allclose(reconstructed, chunk, atol=1e-10)

    def test_all_spectrum_schemas_encode_and_decode(self):
        t = np.linspace(0.0, 1.0, 10)[:, None]
        smooth_arm = np.hstack([np.sin(2.0 * np.pi * t + i * 0.2) for i in range(5)])
        step_gripper = np.where(t >= 0.5, 45.0, 10.0)
        chunk = np.hstack([smooth_arm, step_gripper])

        for schema in (
            "direct",
            "fast_idct",
            "band_idct:0.4",
            "ifft:0.45",
            "idwt_haar",
            "idwt_db4",
            "custom_hybrid",
        ):
            with self.subTest(schema=schema):
                mod = create_spectrum_module(schema, quant_scale=100.0)
                encoded = mod.encode(chunk)
                decoded = mod.decode(encoded)
                self.assertEqual(decoded.shape, (10, 6))
                self.assertTrue(np.all(np.isfinite(decoded)))
                if schema == "direct":
                    np.testing.assert_allclose(decoded, chunk)
                if schema == "custom_hybrid":
                    # Custom hybrid preserves the exact step gripper trace without Gibbs ringing
                    np.testing.assert_allclose(decoded[:, 5], chunk[:, 5], atol=1e-9)

    def test_custom_hybrid_c1_boundary_blending_reduces_jump(self):
        mod = create_spectrum_module("custom_hybrid:0.35", quant_scale=100.0)
        prev_tail = np.zeros((2, 6))
        # Next chunk starts with a sudden offset jump on joint 0
        next_chunk = np.ones((10, 6)) * 0.5
        encoded = mod.encode(next_chunk)
        unblended = mod.decode(encoded, prev_chunk_tail=None)
        blended = mod.decode(encoded, prev_chunk_tail=prev_tail)
        self.assertLess(
            abs(blended[0, 0] - prev_tail[-1, 0]),
            abs(unblended[0, 0] - prev_tail[-1, 0]),
        )

    def test_parse_freq_band(self):
        self.assertEqual(parse_freq_band("0.0,0.35"), (0.0, 0.35))
        self.assertEqual(parse_freq_band("0.4"), (0.0, 0.4))


class MetricsSuiteTests(unittest.TestCase):
    def test_sparc_ldlj_and_hfer_distinguish_smooth_vs_noisy(self):
        t = np.linspace(0.0, 2.0, 60)
        smooth_q = np.column_stack([np.sin(np.pi * t), np.cos(np.pi * t)])
        rng = np.random.default_rng(123)
        noisy_q = smooth_q + 0.35 * np.sin(24.0 * np.pi * t)[:, None] + 0.05 * rng.normal(size=smooth_q.shape)

        smooth_spec = compute_spectral_features(smooth_q, hz=30.0, cutoff_hz=5.0)
        noisy_spec = compute_spectral_features(noisy_q, hz=30.0, cutoff_hz=5.0)
        self.assertLess(
            smooth_spec["high_freq_energy_ratio_hfer"],
            noisy_spec["high_freq_energy_ratio_hfer"],
        )

        smooth_speed = np.linalg.norm(np.gradient(smooth_q, 1.0 / 30.0, axis=0), axis=1)
        noisy_speed = np.linalg.norm(np.gradient(noisy_q, 1.0 / 30.0, axis=0), axis=1)
        # Closer to 0 (less negative) means smoother in SPARC and LDLJ
        self.assertGreater(
            compute_sparc(smooth_speed, hz=30.0),
            compute_sparc(noisy_speed, hz=30.0),
        )
        self.assertGreater(
            compute_ldlj(smooth_q, hz=30.0),
            compute_ldlj(noisy_q, hz=30.0),
        )

    def test_chunk_boundary_jump_computation(self):
        q = np.zeros((25, 3))
        q[10:, 0] = 0.5  # Jump at boundary index 10
        jumps = compute_chunk_boundary_jump(q, chunk_size=10, hz=30.0)
        self.assertGreater(jumps["chunk_boundary_jump_rad_s"], 0.0)


class EndToEndScopesTests(unittest.TestCase):
    def test_scope_1_modular_pipeline_executes_with_spectral_module(self):
        with SO101Simulation(gui=False, training_seed=42) as sim:
            spec = resolve_scene_from_prompt(sim, "Move the apple to the target", seed=42)
        oracle_actions, _, oracle_len = generate_oracle_action_sequence(spec)
        spec.oracle_path_length_m = oracle_len

        with SO101Simulation(gui=False, training_seed=42) as sim:
            apply_scene_spec(sim, spec)
            pipeline = ModularExecutionPipeline(
                simulation=sim,
                policy=PrecomputedChunkPolicy(oracle_actions, chunk_size=10),
                spectrum_module=create_spectrum_module("custom_hybrid:0.45", quant_scale=50.0),
                model_name="astar_oracle",
            )
            result = pipeline.execute_task(spec, max_chunks=25)
            self.assertTrue(result["task_success"])
            self.assertEqual(result["collision_count"], 0.0)
            self.assertIn("joint_rms_jerk_rad_s3", result)
            self.assertIn("sparc_smoothness", result)
            self.assertIn("high_freq_energy_ratio_hfer", result)

    def test_scope_2_and_3_and_4_golden_data_fit_and_eval_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden_dir = root / "golden"
            eval_dir = root / "eval_out"
            # Scope 3: Generate 1 Golden SO-101 episode (seed=9 -> evaluation split)
            manifest = generate_raw_episodes(
                episodes=1, root=golden_dir, start_seed=9, robot="so101"
            )
            self.assertEqual(manifest["episodes"], 1)
            ep_path = golden_dir / "episode_000000"
            meta, arrays = validate_raw_episode(ep_path)
            self.assertEqual(meta["schema_version"], 3)
            self.assertIn("scene_spec", meta)
            self.assertIn("initial_object_poses", meta["scene_spec"])
            self.assertIn("oracle_metrics", meta)
            self.assertIn("dct_coeffs", arrays)

            # Verify action filtering helper in dataset.py
            filtered = filter_episode_actions(arrays["actions"], spectrum_schema="custom_hybrid:0.4")
            self.assertEqual(filtered.shape, arrays["actions"].shape)

            # Scope 4: Fit spectral stats on generated Golden Dataset
            stats = fit_spectral_dataset_stats(
                golden_dir, schemas=("fast_idct", "band_idct:0.35", "custom_hybrid")
            )
            self.assertEqual(stats["robot"], "so101")
            self.assertEqual(len(stats["dct_mode_energy_distribution"]), 10)
            self.assertIn("custom_hybrid", stats["schemas"])
            self.assertEqual(
                resolve_training_spectrum_schema("custom_hybrid", "direct", "0.0,1.0"),
                ("custom_hybrid", (0.0, 0.35)),
            )

            # Scope 2: Run Evaluation Matrix over the Golden Episode
            report = run_evaluation_matrix(
                schemas=("direct", "fast_idct", "custom_hybrid:0.45"),
                model="astar_oracle",
                golden_data=golden_dir,
                split="evaluation",
                episodes=1,
                robot="so101",
                quant_scale=50.0,
                output_dir=eval_dir,
            )
            self.assertEqual(report["num_episodes"], 1)
            self.assertTrue((eval_dir / "eval_metadata.json").exists())
            self.assertTrue((eval_dir / "eval_summary.csv").exists())
            self.assertTrue((eval_dir / "summary_matrix.png").exists())

    def test_webapp_catalog_preview_and_headless_launch(self):
        from smooth_operator.webapp import (
            ARRANGEMENT_CATALOG,
            DECODER_CATALOG,
            build_arrangement_preview,
            execute_web_launch,
            model_catalog,
        )

        models = {m["id"]: m for m in model_catalog()}
        self.assertEqual(models["smolvla"]["kind"], "vla")
        self.assertEqual(models["smolvla"]["variants"]["so101"]["checkpoint"], "lerobot/smolvla_base")
        self.assertEqual(models["smolvla"]["variants"]["libero"]["checkpoint"], "lerobot/smolvla_libero")
        self.assertEqual(models["astar_expert"]["kind"], "scripted")
        for model in models.values():
            for decoder in model["decoders"]:
                self.assertIn(decoder, DECODER_CATALOG)
            # Every model opens on an environment it has a runnable checkpoint for.
            if model["available"]:
                robot = ARRANGEMENT_CATALOG[model["default_arrangement"]]["robot"]
                self.assertTrue(model["variants"][robot]["available"])
        def domains(model_id):
            return {DECODER_CATALOG[d]["domain"] for d in models[model_id]["decoders"]}

        # Each VLA appears only in the tab matching its native action output.
        for model in models.values():
            if model["kind"] == "vla":
                self.assertEqual(domains(model["id"]), {model["domain"]})
        self.assertEqual(domains("smolvla"), {"time_domain"})
        self.assertEqual(domains("pi0fast"), {"frequency_domain"})
        self.assertEqual(set(models["pi0fast"]["decoders"]), {"fast_idct", "band_idct", "custom_hybrid"})
        self.assertEqual(models["pi0fast"]["default_arrangement"], "libero_object")
        self.assertEqual(models["astar_expert"]["robots"], ["so101", "scara"])
        self.assertEqual(domains("astar_expert"), {"time_domain", "frequency_domain"})
        self.assertFalse(models["groot"]["available"])
        self.assertEqual(ARRANGEMENT_CATALOG["libero_object"]["robot"], "libero")
        self.assertEqual(len(ARRANGEMENT_CATALOG["libero_10"]["prompts"]), 10)

        with self.assertRaises(ValueError):
            execute_web_launch({"arrangement_id": "scara_seed_9", "model_id": "smolvla"})
        with self.assertRaises(ValueError):
            execute_web_launch({"arrangement_id": "libero_object", "model_id": "astar_expert"})
        with self.assertRaises(ValueError):
            execute_web_launch({"arrangement_id": "so101_seed_9", "model_id": "groot"})
        with self.assertRaises(ValueError):
            execute_web_launch({"arrangement_id": "so101_seed_9", "model_id": "pi0fast", "decoder": "fast_idct"})
        with self.assertRaises(ValueError):
            execute_web_launch({"arrangement_id": "libero_object", "model_id": "pi0fast", "decoder": "ifft"})

        # Verify arrangement preview renders schematic + live camera views + objects
        preview = build_arrangement_preview("so101_seed_9")
        self.assertIn("schematic_image", preview)
        self.assertTrue(preview["schematic_image"].startswith("data:image/png;base64,"))
        self.assertEqual(len(preview["camera_images"]), 3)
        self.assertGreaterEqual(len(preview["objects"]), 3)
        self.assertGreaterEqual(len(preview["prompts"]), 4)

        # Verify headless launch returns live telemetry and spectral chart
        launch_out = execute_web_launch(
            {
                "arrangement_id": "so101_seed_9",
                "model_id": "astar_expert",
                "decoder": "custom_hybrid",
                "freq_band": "0.0,0.40",
                "quant_scale": 50.0,
                "prompt": "Move the apple to the target",
                "open_gui": False,
                "hold_seconds": 1.0,
            }
        )
        self.assertEqual(launch_out["status"], "ok")
        self.assertTrue(launch_out["metrics"]["task_success"])
        self.assertGreater(launch_out["metrics"]["max_vertical_lift_cm"], 2.0)
        self.assertTrue(launch_out["diagnostic_chart"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
