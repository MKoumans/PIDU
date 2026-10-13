"""
Unit tests for research/analysis/analyze_loss_run.py.

Uses synthetic / mock loss data throughout – no real run files required.
The helpers at the top of this file let you reproduce the same mock data
from the notebook as well.
"""

import io
import sys
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")   # non-interactive backend – no windows opened during tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.analysis.analyze_loss_run import (
    load_loss_data,
    compute_average_statistics,
    create_table,
    load_multiple_runs,
    plot_loss_by_epoch,
    plot_loss_by_epoch_average_only,
    plot_loss_log_scale,
    plot_loss_log_scale_with_runs,
    plot_model_runs_log_scale,
    plot_loss_by_epoch_with_runs,
)

from research.legacy.analyse_hyperparameters import extract_hyperparameter_info
from pidu.paths import case_dir


# ─────────────────────────────────────────────────────────────────────────────
# Mock-data factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_df(
    n_epochs: int = 5,
    n_train_batches: int = 8,
    n_val_batches: int = 2,
    n_test_batches: int = 2,
    n_exp_batches: int = 1,
    run_id: int = 0,
    seed: int = 0,
    start_loss: float = 10_000.0,
    decay: float = 0.6,
) -> pd.DataFrame:
    """Return a DataFrame in the format produced by load_loss_data, with a
    run_id column added.  Loss decays geometrically across epochs plus noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    t = 0.0

    for epoch in range(n_epochs):
        epoch_loss_scale = start_loss * (decay ** epoch)

        for batch in range(n_train_batches):
            loss = epoch_loss_scale * rng.uniform(0.8, 1.2)
            rows.append(("train", 0, batch, epoch, loss, t))
            t += 0.5

        for batch in range(n_val_batches):
            loss = epoch_loss_scale * rng.uniform(0.85, 1.15)
            rows.append(("val", 0, batch, epoch, loss, t))
            t += 0.3

        for batch in range(n_test_batches):
            loss = epoch_loss_scale * rng.uniform(0.85, 1.15)
            rows.append(("test", 0, batch, epoch, loss, t))
            t += 0.3

        for batch in range(n_exp_batches):
            loss = epoch_loss_scale * rng.uniform(0.7, 1.0)
            rows.append(("exp", 0, batch, epoch, loss, t))
            t += 0.2

    df = pd.DataFrame(rows, columns=["dataset", "system", "batch", "epoch", "loss", "time"])
    df["run_id"] = run_id
    return df


def _make_combined_df(n_runs: int = 4, **kwargs) -> pd.DataFrame:
    """Return a combined DataFrame from N independent runs."""
    return pd.concat(
        [_make_mock_df(run_id=i, seed=i, **kwargs) for i in range(n_runs)],
        ignore_index=True,
    )


def _write_mock_run_file(path: Path, n_epochs: int = 5, n_header_lines: int = 13,
                         seed: int = 0) -> None:
    """Write a mock run file (header + CSV body) to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _make_mock_df(n_epochs=n_epochs, seed=seed)
    df = df.drop(columns=["run_id"])   # real files don't have run_id
    header = "\n".join([f"meta_line_{i}: value" for i in range(n_header_lines)])
    with open(path, "w") as fh:
        fh.write(header + "\n")
        df.to_csv(fh, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_loss_data
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadLossData(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        plt.close("all")

    def _write_csv(self, rows: str, filename: str = "test.txt") -> Path:
        p = self.tmp / filename
        p.write_text(rows)
        return p

    def test_load_no_skiprows(self):
        p = self._write_csv("dataset,system,batch,epoch,loss,time\ntrain,0,0,0,100.0,0.5\n")
        df = load_loss_data(p)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["loss"].iloc[0], 100.0)

    def test_load_with_skiprows(self):
        header = "meta1\nmeta2\nmeta3\n"
        body = "dataset,system,batch,epoch,loss,time\ntrain,0,0,0,200.0,0.5\nval,0,0,0,180.0,0.3\n"
        p = self._write_csv(header + body)
        df = load_loss_data(p, skiprows=3)
        self.assertEqual(len(df), 2)
        self.assertIn("dataset", df.columns)
        self.assertIn("loss", df.columns)

    def test_expected_columns_present(self):
        p = self._write_csv("dataset,system,batch,epoch,loss,time\ntrain,0,0,0,1.0,0.1\n")
        df = load_loss_data(p)
        for col in ("dataset", "system", "batch", "epoch", "loss", "time"):
            self.assertIn(col, df.columns, f"missing column: {col}")

    def test_mock_file_roundtrip(self):
        path = self.tmp / "loss_PIDU_0_000.txt"
        _write_mock_run_file(path, n_epochs=3, n_header_lines=13)
        df = load_loss_data(path, skiprows=13)
        self.assertIn("dataset", df.columns)
        self.assertTrue(len(df) > 0)
        self.assertTrue((df["loss"] > 0).all())


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_average_statistics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAverageStatistics(unittest.TestCase):
    def setUp(self):
        self.df = _make_combined_df(n_runs=4, n_epochs=5)

    def test_output_columns(self):
        result = compute_average_statistics(self.df)
        for col in ("dataset", "epoch", "loss_mean", "loss_std", "loss_min", "loss_max"):
            self.assertIn(col, result.columns, f"missing: {col}")

    def test_all_datasets_present(self):
        result = compute_average_statistics(self.df)
        for ds in ("train", "val", "test", "exp"):
            self.assertIn(ds, result["dataset"].values, f"missing dataset: {ds}")

    def test_mean_within_range(self):
        result = compute_average_statistics(self.df)
        train_means = result[result["dataset"] == "train"]["loss_mean"]
        # All means should be positive and finite
        self.assertTrue((train_means > 0).all())
        self.assertFalse(train_means.isna().any())

    def test_decreasing_loss(self):
        # Mean train loss should decrease over epochs (geometric decay mock)
        result = compute_average_statistics(self.df)
        train = result[result["dataset"] == "train"].sort_values("epoch")
        first = train["loss_mean"].iloc[0]
        last = train["loss_mean"].iloc[-1]
        self.assertGreater(first, last)

    def test_std_non_negative(self):
        result = compute_average_statistics(self.df)
        self.assertTrue((result["loss_std"].fillna(0) >= 0).all())

    def test_single_run_no_crash(self):
        single = _make_mock_df(n_epochs=3)
        result = compute_average_statistics(single)
        self.assertIsInstance(result, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: create_table
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateTable(unittest.TestCase):
    def setUp(self):
        self.df = _make_combined_df(n_runs=4, n_epochs=5)
        self.method = "PIDU"
        self.train, self.val, self.test, self.exp = create_table(self.df, self.method)

    def test_returns_four_dicts(self):
        result = create_table(self.df, self.method)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_column_names_contain_method(self):
        for key in self.train.keys():
            self.assertIn(self.method, key)

    def test_mean_and_std_keys_exist(self):
        self.assertIn(f"{self.method} mean (train)", self.train)
        self.assertIn(f"{self.method} std (train)", self.train)

    def test_lengths_equal_n_epochs(self):
        n_epochs = self.df["epoch"].nunique()
        mean_arr = self.train[f"{self.method} mean (train)"]
        self.assertEqual(len(mean_arr), n_epochs)

    def test_exp_dict_populated(self):
        self.assertIn(f"{self.method} mean (exp)", self.exp)

    def test_no_exp_data_empty_dict(self):
        df_no_exp = self.df[self.df["dataset"] != "exp"].copy()
        _, _, _, exp = create_table(df_no_exp, "DU")
        self.assertEqual(len(exp), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_multiple_runs  (writes temp files on disk)
# ─────────────────────────────────────────────────────────────────────────────

_TEST_USECASE = 999   # a usecase number that won't clash with real data


class TestLoadMultipleRuns(unittest.TestCase):
    """Creates temporary run files inside <case dir>/runs/ (a temporary folder, see conftest.py) and
    removes them when the test finishes."""

    def setUp(self):
        self.runs_dir = case_dir(_TEST_USECASE) / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._written: list[Path] = []

    def tearDown(self):
        for p in self._written:
            p.unlink(missing_ok=True)
        # Remove the temporary case directory if empty
        try:
            shutil.rmtree(self.runs_dir.parent)
        except Exception:
            pass
        plt.close("all")

    def _write(self, pattern: str, run_idx: int, trail: int, n_epochs: int = 3,
               seed: int = 0) -> Path:
        name = f"{pattern}_{run_idx}_{trail:03d}.txt"
        p = self.runs_dir / name
        _write_mock_run_file(p, n_epochs=n_epochs, n_header_lines=13, seed=seed)
        self._written.append(p)
        return p

    def test_load_two_runs(self):
        pattern = "loss_PIDU"
        trail = 0
        self._write(pattern, 0, trail, seed=0)
        self._write(pattern, 1, trail, seed=1)
        combined, individual = load_multiple_runs(
            pattern, trail, num_runs=2, usecase=_TEST_USECASE
        )
        self.assertIn("run_id", combined.columns)
        self.assertEqual(set(combined["run_id"].unique()), {0, 1})
        self.assertEqual(len(individual), 2)

    def test_combined_df_has_correct_datasets(self):
        pattern = "loss_DU"
        trail = 1
        self._write(pattern, 0, trail, seed=2)
        combined, _ = load_multiple_runs(pattern, trail, num_runs=1, usecase=_TEST_USECASE)
        for ds in ("train", "val", "test", "exp"):
            self.assertIn(ds, combined["dataset"].values)

    def test_missing_runs_skipped_gracefully(self):
        """If only 1 of 3 files exists, the function should still succeed."""
        pattern = "loss_CNN"
        trail = 2
        self._write(pattern, 1, trail, seed=3)   # only run_idx=1 exists
        combined, individual = load_multiple_runs(
            pattern, trail, num_runs=3, usecase=_TEST_USECASE
        )
        self.assertEqual(len(individual), 1)
        self.assertEqual(set(combined["run_id"].unique()), {1})

    def test_all_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_multiple_runs("loss_NONEXISTENT", 99, num_runs=2, usecase=_TEST_USECASE)

    def test_run_id_reflects_file_index(self):
        pattern = "loss_PIDU"
        trail = 3
        for i in range(3):
            self._write(pattern, i, trail, seed=i)
        combined, _ = load_multiple_runs(pattern, trail, num_runs=3, usecase=_TEST_USECASE)
        self.assertEqual(sorted(combined["run_id"].unique()), [0, 1, 2])


# ─────────────────────────────────────────────────────────────────────────────
# Tests: per-pattern trail (the new (pattern, trail) tuple feature)
# ─────────────────────────────────────────────────────────────────────────────

class TestPerPatternTrail(unittest.TestCase):
    """Verify that main() dispatches the right trail for each pattern."""

    def setUp(self):
        self.runs_dir_base = case_dir(_TEST_USECASE) / "runs"
        self.runs_dir_base.mkdir(parents=True, exist_ok=True)
        self._written: list[Path] = []

    def tearDown(self):
        for p in self._written:
            p.unlink(missing_ok=True)
        try:
            shutil.rmtree(self.runs_dir_base.parent)
        except Exception:
            pass
        plt.close("all")

    def _write(self, pattern: str, run_idx: int, trail: int, seed: int = 0) -> Path:
        name = f"{pattern}_{run_idx}_{trail:03d}.txt"
        p = self.runs_dir_base / name
        _write_mock_run_file(p, n_epochs=3, n_header_lines=13, seed=seed)
        self._written.append(p)
        return p

    def test_string_entry_uses_global_trail(self):
        """A plain string in base_patterns should use the global trail."""
        self._write("loss_PIDU_nowgn_nostruct", 0, 20, seed=0)
        from research.analysis.analyze_loss_run import main
        # Should NOT raise – file with trail=20 exists and global trail=20
        main(
            base_patterns=["loss_PIDU_nowgn_nostruct"],
            trail=20,
            usecase=_TEST_USECASE,
            num_runs=1,
            compare=False,
            export_plots=False,
        )

    def test_tuple_overrides_global_trail(self):
        """A (pattern, trail) tuple should override the global trail."""
        self._write("loss_PIDU_wgn_nostruct", 0, 21, seed=1)
        from research.analysis.analyze_loss_run import main
        # global trail=0 would look for trail 000 which doesn't exist;
        # the tuple (pattern, 21) should find the file with 021
        main(
            base_patterns=[("loss_PIDU_wgn_nostruct", 21)],
            trail=0,   # global trail = 0 (wrong file)
            usecase=_TEST_USECASE,
            num_runs=1,
            compare=False,
            export_plots=False,
        )

    def test_mixed_string_and_tuple(self):
        """Mix plain strings and tuples in the same call."""
        self._write("loss_PIDU_nowgn_nostruct", 0, 20, seed=0)
        self._write("loss_PIDU_wgn_struct",     0, 22, seed=2)
        from research.analysis.analyze_loss_run import main
        main(
            base_patterns=[
                ("loss_PIDU_nowgn_nostruct", 20),
                ("loss_PIDU_wgn_struct", 22),
            ],
            trail=0,
            usecase=_TEST_USECASE,
            num_runs=1,
            compare=False,
            export_plots=False,
        )

    def test_cli_colon_syntax_parsed(self):
        """Verify the CLI colon pattern:trail parser matches the tuple approach."""
        def _parse_pattern(s: str):
            if ":" in s:
                pat, t = s.rsplit(":", 1)
                return (pat, int(t))
            return s

        self.assertEqual(_parse_pattern("loss_PIDU_nowgn_nostruct:20"),
                         ("loss_PIDU_nowgn_nostruct", 20))
        self.assertEqual(_parse_pattern("loss_PIDU_wgn_nostruct:21"),
                         ("loss_PIDU_wgn_nostruct", 21))
        self.assertEqual(_parse_pattern("loss_PIDU"),
                         "loss_PIDU")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: extract_hyperparameter_info
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractHyperparameterInfo(unittest.TestCase):
    def test_simple_pattern(self):
        info = extract_hyperparameter_info("loss_PIDU_5_002.txt")
        self.assertEqual(info["model"], "PIDU")
        self.assertEqual(info["run_number"], "002")

    def test_hyperparameter_extraction(self):
        info = extract_hyperparameter_info("loss_PIDU_initialiser_-kaiming__6_006.txt")
        self.assertEqual(info["model"], "PIDU")
        self.assertIn("initialiser", info["hyperparameters"])
        self.assertEqual(info["hyperparameters"]["initialiser"], "kaiming")

    def test_scientific_notation_decoded(self):
        info = extract_hyperparameter_info("loss_PIDU_lr_-1e_3__2_003.txt")
        self.assertEqual(info["hyperparameters"].get("lr"), "1e-3")

    def test_no_hyperparameters_default_config(self):
        info = extract_hyperparameter_info("loss_DU_3_001.txt")
        self.assertEqual(info["config_key"], "DU__default")

    def test_no_loss_prefix_returns_unknown(self):
        info = extract_hyperparameter_info("random_file_name.txt")
        self.assertEqual(info["model"], "unknown")

    def test_config_label_contains_model(self):
        info = extract_hyperparameter_info("loss_CNN_0_000.txt")
        self.assertIn("CNN", info["config_label"])

    def test_multiple_hyperparameters(self):
        info = extract_hyperparameter_info(
            "loss_PIDU_mixed_precision_-on__lr_-1e_3__2_003.txt"
        )
        hparams = info["hyperparameters"]
        self.assertIn("mixed_precision", hparams)
        self.assertIn("lr", hparams)
        self.assertEqual(hparams["lr"], "1e-3")
        self.assertEqual(hparams["mixed_precision"], "on")

    def test_wgn_nostruct_pattern(self):
        """Patterns like loss_PIDU_wgn_nostruct_XX_021 have no _- separator,
        so they fall through to the legacy variant path."""
        info = extract_hyperparameter_info("loss_PIDU_wgn_nostruct_0_021.txt")
        self.assertEqual(info["model"], "PIDU")
        self.assertEqual(info["run_number"], "021")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: plotting smoke tests (just verify no exception + correct return type)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotSmoke(unittest.TestCase):
    def setUp(self):
        self.single = _make_mock_df(n_epochs=5, run_id=0)
        self.multi = _make_combined_df(n_runs=4, n_epochs=5)

    def tearDown(self):
        plt.close("all")

    def test_plot_loss_by_epoch(self):
        fig, ax = plot_loss_by_epoch(self.single)
        self.assertIsNotNone(fig)

    def test_plot_loss_log_scale(self):
        fig, ax = plot_loss_log_scale(self.single)
        self.assertIsNotNone(fig)

    def test_plot_loss_by_epoch_average_only(self):
        fig, ax = plot_loss_by_epoch_average_only(self.multi)
        self.assertIsNotNone(fig)

    def test_plot_loss_by_epoch_with_runs(self):
        fig, axes = plot_loss_by_epoch_with_runs(self.multi)
        self.assertIsNotNone(fig)

    def test_plot_loss_log_scale_with_runs(self):
        fig, ax = plot_loss_log_scale_with_runs(self.multi)
        self.assertIsNotNone(fig)

    def test_plot_model_runs_log_scale_test(self):
        fig, ax = plot_model_runs_log_scale(self.multi, dataset_type="test",
                                            model_name="PIDU")
        self.assertIsNotNone(fig)

    def test_plot_model_runs_log_scale_val(self):
        fig, ax = plot_model_runs_log_scale(self.multi, dataset_type="val",
                                            model_name="DU")
        self.assertIsNotNone(fig)

    def test_plot_save_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "test_plot.png"
            plot_loss_by_epoch(self.single, save_path=save_path)
            self.assertTrue(save_path.exists())

    def test_plot_model_runs_no_run_id_raises(self):
        df_no_run = self.multi.drop(columns=["run_id"])
        with self.assertRaises(ValueError):
            plot_model_runs_log_scale(df_no_run)

    def test_plot_model_runs_wrong_dataset_raises(self):
        with self.assertRaises(ValueError):
            plot_model_runs_log_scale(self.multi, dataset_type="nonexistent")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
