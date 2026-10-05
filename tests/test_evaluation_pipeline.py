"""Every evaluation script, end to end, on a tiny synthetic cohort (no patient data)."""
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from evaluation import apply_external, evaluate, history_ablation, make_synthetic, train


def test_synthetic_end_to_end(tmp_path, monkeypatch):
    data, runs, out = tmp_path / "data", tmp_path / "runs", tmp_path / "out"
    make_synthetic.main(["--out", str(data), "--splits", "2", "--patients", "160",
                         "--external-patients", "60", "--seed", "1"])
    monkeypatch.setenv("DDH_DATA_ROOT", str(data))
    monkeypatch.setenv("DDH_RUNS_ROOT", str(runs))
    monkeypatch.setenv("DDH_OUTPUT_DIR", str(out))
    monkeypatch.setenv("DDH_COLUMNS", str(data / "columns.json"))

    common = ["--splits", "0", "1", "--iters", "2", "--batch-size", "64", "--all-landmark-nll",
              "--alpha", "0", "--hazard"]
    train.main(["--arm", "patient", *common])
    train.main(["--arm", "band", *common, "--landmark-row-weighting", "band"])
    for arm in ("patient", "band"):
        apply_external.main(["--arm", arm, "--splits", "0", "1"])
    evaluate.main(["--arm", "band", "--arm", "1/V_i=patient", "--splits", "0", "1",
                   "--horizons", "90", "--n-boot", "20"])
    history_ablation.main(["--arm", "band", "--splits", "0", "1", "--horizons", "90",
                           "--gap-col", "days_since_last_obs", "--n-boot", "20"])

    landmarks = pd.read_csv(out / "landmarks_summary.csv")
    assert set(landmarks.arm) == {"band", "1/V_i"}
    assert landmarks["pred"].between(0, 1).all()
    external = pd.read_csv(out / "external.csv")
    assert {"auc_delta_vs_ref", "auc_delta_lo", "auc_delta_hi"} <= set(external.columns)
    assert external.loc[external.arm == "band", "auc_delta_vs_ref"].isna().all()
    assert set(pd.read_csv(out / "ablation_summary.csv").variant) == {"full", "masked", "reanchored"}
    for split in (0, 1):
        assert (runs / "band" / f"split_{split}" / "metadata.json").exists()
