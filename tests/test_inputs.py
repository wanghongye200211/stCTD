from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stctd import DataConfig  # noqa: E402
from stctd.data import load_prepared_data  # noqa: E402

spec = importlib.util.spec_from_file_location("train_cli", REPO_ROOT / "examples" / "train_stctd.py")
train_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_cli)


class PreparedInputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.h5ad"
        self.adata = ad.AnnData(
            X=np.ones((4, 3), dtype=np.float32),
            obs=pd.DataFrame({"stage": [0, 0, 1, 1], "day": [2, 2, 4, 4], "type": ["A", "B", "A", "B"]}, index=list("abcd")),
        )
        self.adata.obsm["xy"] = np.arange(8, dtype=np.float32).reshape(4, 2)
        self.adata.obsm["z"] = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.config = DataConfig(
            input_h5ad=self.path,
            network_tsv=Path("unused.tsv"),
            gat_out_dir=Path(self.temp.name),
            use_precomputed_embedding=True,
            spatial_key="xy",
            latent_key="z",
            time_key="stage",
            raw_time_key="day",
            annotation_key="type",
        )

    def load(self):
        self.adata.write_h5ad(self.path)
        return load_prepared_data(self.config, torch.device("cpu"))[0]

    def test_custom_keys_and_expression_fallback_roundtrip(self) -> None:
        data = self.load()
        self.assertEqual(data.time_values, [0.0, 1.0])
        self.assertEqual(data.raw_time_values, [2.0, 4.0])
        spatial, latent = data.scaler.inverse(torch.cat(data.state_by_time).numpy())
        np.testing.assert_allclose(spatial, self.adata.obsm["xy"], atol=1e-6)
        np.testing.assert_allclose(latent, self.adata.obsm["z"], atol=1e-6)
        np.testing.assert_array_equal(torch.cat(data.expression_by_time).numpy(), self.adata.X)

    def test_missing_field_has_actionable_error(self) -> None:
        del self.adata.obsm["z"]
        with self.assertRaisesRegex(ValueError, "obsm\\['z'\\]"):
            self.load()

    def test_disabled_scaling_ignores_zero_spatial_weight(self) -> None:
        self.config = replace(self.config, scale_features=False, spatial_weight=0.0)
        data = self.load()
        np.testing.assert_array_equal(torch.cat(data.state_by_time).numpy()[:, :2], self.adata.obsm["xy"])
        self.assertEqual(data.scaler.spatial_weight, 1.0)

    def test_rejects_unsupported_spatial_dimension(self) -> None:
        self.adata.obsm["xy"] = np.zeros((4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "two spatial columns"):
            self.load()

    def test_rejects_nonfinite_latent(self) -> None:
        self.adata.obsm["z"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            self.load()

    def test_rejects_single_time_point(self) -> None:
        self.adata.obs["stage"] = 0
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.load()


class TrainingCliTest(unittest.TestCase):
    def test_invalid_combinations_fail_before_training(self) -> None:
        for options in [
            ["--epochs", "0"],
            ["--n-hidden", "-1"],
            ["--sample-growth-step", "-1"],
            ["--spatial-weight", "0"],
            ["--no-scale-features", "--spatial-weight", "nan"],
            ["--no-scale-features", "--spatial-weight", "inf"],
            ["--ode-step-size", "nan"],
            ["--use-cell-type-prior"],
            ["--lambda-ssp", "1"],
            ["--lambda-action", "1"],
            ["--velocity-parameterization", "vector", "--lambda-hj", "1"],
        ]:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                train_cli.parse_args(["--input-h5ad", str(Path(__file__)), *options])
            self.assertEqual(exc.exception.code, 2)

    def test_supported_zero_values_remain_available(self) -> None:
        args = train_cli.parse_args([
            "--input-h5ad", str(Path(__file__)),
            "--n-hidden", "0", "--sample-growth-step", "0",
            "--no-scale-features", "--spatial-weight", "0",
        ])
        self.assertEqual(args.n_hidden, 0)
        self.assertEqual(args.sample_growth_step, 0)
        self.assertFalse(args.scale_features)
        self.assertEqual(args.spatial_weight, 0.0)

    def test_baseline_accepts_current_and_legacy_summaries(self) -> None:
        metrics = [{"label_prop_corr": 1, "label_prop_l1": 0, "spatial_chamfer_norm": 0.1, "spatial_grid_iou": 0.5}]
        summary = {"run_dir": "current", "config": {"method": "stCTD"}, "evaluation": {"metrics": metrics}}
        with tempfile.TemporaryDirectory() as temp:
            nested = Path(temp) / "current.json"
            legacy = Path(temp) / "legacy.json"
            nested.write_text(json.dumps(summary))
            legacy.write_text(json.dumps({"metrics": metrics, "config": {"lambda_hj": 10}}))
            rows = train_cli._metrics_rows(summary, [nested, legacy])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["method"], "stCTD")
        self.assertEqual(rows[2]["method"], "baseline lambda_hj=10")


if __name__ == "__main__":
    unittest.main()
