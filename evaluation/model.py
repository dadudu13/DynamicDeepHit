"""Build, train, save, load and score a DynamicDeepHit model for one split.

Predictions are always made from a landmark s: the raw model output F_k(t) is a
CIF from day 0, and the quantity a clinician needs at day s is

    F_k(s + h | T > s) = (F_k(s + h) - F_k(s)) / S(s),   S(s) = 1 - sum_k F_k(s)

for each horizon h. condition_cif_on_landmark and predict_at_horizons implement
exactly that; every evaluation table is built from their output.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ddh import DynamicDeepHit

from .config import (CHECKPOINT_FILE, COMPETING_EVENT, EVENT_OF_INTEREST, METADATA_FILE,
                     Columns)
from .data import SequenceDataset, choose_time_grid


@dataclass
class Architecture:
    """Network shape. Stored with every run so a checkpoint can be rebuilt exactly."""

    hidden_rnn: int = 32
    layers_rnn: int = 1
    typ: str = "GRU"
    long_layers: list[int] = field(default_factory=lambda: [64])
    att_layers: list[int] = field(default_factory=lambda: [32])
    cs_layers: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    embedding_dim: int = 16
    hazard: bool = False
    proportional: bool = False


@dataclass
class Objective:
    """Training objective. Defaults reproduce the library's original loss."""

    alpha: float = 0.01                  # ranking term
    beta: float = 0.75                   # survival NLL; 1 - alpha - beta goes to reconstruction
    sigma: float = 0.5                   # ranking sharpness
    all_landmark_nll: bool = False       # NLL at every (patient, visit) landmark
    landmark_row_weighting: str = "patient"   # patient (1/V_i) | visit | band
    landmark_band_edges: list[float] | None = None
    landmark_decay_days: float | None = None
    landmark_augment_prob: float = 0.0
    gamma: float = 0.0                   # horizon-specific likelihood term
    horizon_days: list[int] | None = None
    horizon_ipcw_clip: float = 10.0
    first_visit_only: bool = False


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def categorical_embedding(train: pd.DataFrame, features: list[str], cols: Columns) -> tuple[int | None, int | None]:
    """(vocabulary size, column index) for the embedded column, or (None, None)."""
    if cols.categorical is None or cols.categorical not in features:
        return None, None
    return int(train[cols.categorical].max()) + 1, features.index(cols.categorical)


def build_model(split_time: list[float], arch: Architecture, objective: Objective,
                vocab_size: int | None, categorical_index: int | None) -> DynamicDeepHit:
    embed = vocab_size is not None
    return DynamicDeepHit(
        split=split_time, layers_rnn=arch.layers_rnn, hidden_rnn=arch.hidden_rnn, typ=arch.typ,
        long_param={"layers": arch.long_layers, "dropout": arch.dropout},
        att_param={"layers": arch.att_layers, "dropout": arch.dropout},
        cs_param={"layers": arch.cs_layers, "dropout": arch.dropout},
        alpha=objective.alpha, beta=objective.beta, sigma=objective.sigma, cuda=False,
        cancer_vocab_size=vocab_size,
        cancer_embedding_dim=arch.embedding_dim if embed else None,
        cancer_id_col=categorical_index,
        hazard=arch.hazard, proportional=arch.proportional,
    )


def train(train: SequenceDataset, val: SequenceDataset, horizons: list[int],
          arch: Architecture, objective: Objective, vocab_size: int | None,
          categorical_index: int | None, seed: int = 42, iters: int = 64,
          batch_size: int = 256, learning_rate: float = 1e-3, patience_limit: int = 10,
          grad_clip_norm: float = 1.0) -> DynamicDeepHit:
    """Fit on one split. The likelihood is conditioned on each patient's landmark s."""
    set_determinism(seed)
    split_time = choose_time_grid(train, val, horizons)
    model = build_model(split_time, arch, objective, vocab_size, categorical_index)
    needs_days = objective.all_landmark_nll or objective.landmark_augment_prob > 0
    model.fit(
        train.x, train.t, train.e,
        val_data=(val.x, val.t, val.e),
        iters=iters, learning_rate=learning_rate, batch_size=batch_size, random_state=seed,
        s=train.last_obs_days, s_val=val.last_obs_days,
        patience_limit=patience_limit, grad_clip_norm=grad_clip_norm,
        visit_days=train.visit_days if needs_days else None,
        visit_days_val=val.visit_days if objective.all_landmark_nll else None,
        landmark_augment_prob=objective.landmark_augment_prob,
        all_landmark_nll=objective.all_landmark_nll,
        landmark_decay_days=objective.landmark_decay_days,
        landmark_row_weighting=objective.landmark_row_weighting,
        landmark_band_edges=objective.landmark_band_edges,
        gamma=objective.gamma,
        horizon_days=list(objective.horizon_days) if objective.horizon_days else None,
        horizon_ipcw_clip=objective.horizon_ipcw_clip,
        event_of_interest=EVENT_OF_INTEREST - 1,
    )
    return model


# ---------------------------------------------------------------- prediction
def predict_cif_curves(model: DynamicDeepHit, ds: SequenceDataset, risk: int,
                       bs: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Unconditional CIF of cause `risk` on the model's time grid, one row per sample."""
    grid = np.asarray(model.split_time[1:], dtype=float)
    curves = []
    with torch.no_grad():
        for start in range(0, len(ds.x), bs):
            batch = ds.x[start:start + bs]
            if len(batch) == 0:
                continue
            _, outcomes = model.torch_model(model._prepocess_test_data(batch))
            curves.append(torch.cumsum(outcomes[risk - 1], dim=1).cpu().numpy())
    if not curves:
        return np.empty((0, len(grid))), grid
    return np.concatenate(curves, axis=0), grid


def condition_cif_on_landmark(cif_1: np.ndarray, cif_2: np.ndarray, grid: np.ndarray,
                              s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F_k(t | T > s) = (F_k(t) - F_k(s)) / S(s) for both causes.

    Samples with s before the first grid point keep the raw CIF (exact at s = 0;
    the model has no resolution below the first point), and so do samples with s
    at or past the last grid point, where the conditional CIF is undefined.
    """
    s_idx = np.clip(np.searchsorted(grid, s, side="right") - 1, 0, len(grid) - 1)
    rows = np.arange(len(cif_1))
    f1_s, f2_s = cif_1[rows, s_idx], cif_2[rows, s_idx]
    surv_s = np.clip(1.0 - f1_s - f2_s, 1e-8, 1.0)
    c1 = np.clip((cif_1 - f1_s[:, None]) / surv_s[:, None], 0.0, 1.0)
    c2 = np.clip((cif_2 - f2_s[:, None]) / surv_s[:, None], 0.0, 1.0)
    raw = (s < grid[0]) | (s >= grid[-1])
    c1[raw], c2[raw] = cif_1[raw], cif_2[raw]
    return c1, c2


def predict_at_horizons(model: DynamicDeepHit, ds: SequenceDataset,
                        horizons: list[int], cols: Columns) -> pd.DataFrame:
    """Risk of each cause by s + h, conditioned on surviving to s, for every sample.

    Adds risk_{h}d (event of interest), competing_risk_{h}d and survival_{h}d to
    the sample index, with entry_time = s and time_ref = time remaining after s.
    Samples whose s is at or past the model's grid get NaN risks and
    beyond_model_horizon = True, so they cannot slip into a metric.
    """
    cif_1, grid = predict_cif_curves(model, ds, EVENT_OF_INTEREST)
    cif_2, _ = predict_cif_curves(model, ds, COMPETING_EVENT)
    cif_1, cif_2 = condition_cif_on_landmark(cif_1, cif_2, grid, ds.last_obs_days)
    at = ds.last_obs_days[:, None] + np.asarray(horizons, dtype=float)[None, :]
    r1 = np.vstack([np.interp(at[i], grid, cif_1[i], left=0.0, right=float(cif_1[i, -1]))
                    for i in range(len(cif_1))]) if len(cif_1) else np.empty((0, len(horizons)))
    r2 = np.vstack([np.interp(at[i], grid, cif_2[i], left=0.0, right=float(cif_2[i, -1]))
                    for i in range(len(cif_2))]) if len(cif_2) else np.empty((0, len(horizons)))

    out = ds.sample_index.copy()
    out["entry_time"] = ds.last_obs_days.astype(int)
    out["time_ref"] = (out[cols.time] - ds.last_obs_days).clip(lower=0)
    for j, h in enumerate(horizons):
        out[f"risk_{h}d"] = r1[:, j]
        out[f"competing_risk_{h}d"] = r2[:, j]
        out[f"survival_{h}d"] = np.clip(1.0 - r1[:, j] - r2[:, j], 0.0, 1.0)
    beyond = ds.last_obs_days >= grid[-1]
    out["beyond_model_horizon"] = beyond
    if beyond.any():
        risk_cols = [f"{p}_{h}d" for p in ("risk", "competing_risk", "survival") for h in horizons]
        out.loc[beyond, risk_cols] = np.nan
    return out


# ---------------------------------------------------------------- persistence
def file_fingerprint(paths: dict[str, Path]) -> dict[str, Any]:
    """Content hash of each input file: two runs on different data cannot be compared silently."""
    out = {}
    for name, path in paths.items():
        if not Path(path).exists():
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        out[name] = {"file": Path(path).name, "sha256": digest.hexdigest()[:16],
                     "bytes": Path(path).stat().st_size}
    return out


def environment() -> dict[str, Any]:
    from importlib import metadata as im
    versions = {}
    for pkg in ("torch", "numpy", "pandas", "scikit-learn", "statsmodels", "scipy"):
        try:
            versions[pkg] = im.version(pkg)
        except Exception:
            versions[pkg] = None
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": versions, "code_commit": commit}


def save_run(run_dir: Path, model: DynamicDeepHit, *, predictors: list[str], horizons: list[int],
             arch: Architecture, objective: Objective, cols: Columns, vocab_size: int | None,
             categorical_index: int | None, inputs: dict[str, Path], seed: int,
             predictions: dict[str, pd.DataFrame]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.torch_model.state_dict(), run_dir / CHECKPOINT_FILE)
    for name, frame in predictions.items():
        frame.to_csv(run_dir / name, index=False)
    meta = {
        "predictors": predictors, "horizons": horizons,
        "split_time": [float(v) for v in model.split_time],
        "architecture": asdict(arch), "objective": asdict(objective), "columns": cols.to_dict(),
        "categorical_vocab_size": vocab_size, "categorical_index": categorical_index,
        "seed": seed, "inputs": file_fingerprint(inputs), "environment": environment(),
    }
    (run_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2, default=_json_default))


def load_run(run_dir: Path, cols: Columns, train_frame: pd.DataFrame | None = None) -> tuple[DynamicDeepHit, dict]:
    """Rebuild the architecture recorded in a run's metadata and load its weights.

    Runs written by older pipelines carry the objective under
    validations.training_objective and no architecture block; those fall back to
    the defaults above, and the categorical vocabulary is recomputed from the
    split's training table.
    """
    meta = json.loads((run_dir / METADATA_FILE).read_text())
    predictors = meta["predictors"]
    legacy = meta.get("validations", {}).get("training_objective", {})
    arch = Architecture(**meta["architecture"]) if "architecture" in meta else Architecture(
        hidden_rnn=int(legacy.get("hidden_rnn") or 32), hazard=bool(legacy.get("hazard")),
        proportional=bool(legacy.get("proportional")))
    objective = Objective(**meta["objective"]) if "objective" in meta else Objective()
    if "categorical_vocab_size" in meta:
        vocab, index = meta["categorical_vocab_size"], meta["categorical_index"]
    elif train_frame is not None:
        vocab, index = categorical_embedding(train_frame, predictors, cols)
    else:
        raise ValueError(f"{run_dir}: metadata has no categorical vocabulary; pass the split's train table")
    model = build_model(meta["split_time"], arch, objective, vocab, index)
    model.torch_model = model._gen_torch_model(len(predictors), optimizer="Adam", risks=2)
    model.torch_model.load_state_dict(torch.load(run_dir / _checkpoint_name(run_dir), map_location="cpu"))
    model.torch_model.eval()
    model.fitted = True
    return model, meta


def _checkpoint_name(run_dir: Path) -> str:
    # Older runs named the checkpoint ddh_model_state.pt.
    return CHECKPOINT_FILE if (run_dir / CHECKPOINT_FILE).exists() else "ddh_model_state.pt"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
