"""Compatibility imports for DSM dependencies."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


_PACKAGE_NAME = "auton_survival"
_SOURCE_DIR_NAME = "auton-survival"
_ENV_ROOT = "AUTON_SURVIVAL_ROOT"


def _is_auton_survival_root(path: Path) -> bool:
    return path.is_dir() and (path / _PACKAGE_NAME).is_dir()


def _candidate_source_roots() -> list[Path]:
    env_root = os.environ.get(_ENV_ROOT)
    roots = []

    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not _is_auton_survival_root(root):
            raise FileNotFoundError(
                f"{_ENV_ROOT} must point to an auton-survival source root "
                f"containing '{_PACKAGE_NAME}/': {root}"
            )
        roots.append(root)

    for parent in Path(__file__).resolve().parents:
        root = parent / _SOURCE_DIR_NAME
        if _is_auton_survival_root(root):
            roots.append(root)

    return roots


def _ensure_auton_survival_available() -> str:
    for root in _candidate_source_roots():
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return root_str

    if importlib.util.find_spec(_PACKAGE_NAME) is not None:
        return "installed"

    raise ModuleNotFoundError(
        "Could not locate auton-survival. Install the 'auton-survival' package, "
        f"place an '{_SOURCE_DIR_NAME}' checkout near DynamicDeepHit, or set "
        f"{_ENV_ROOT} to its source root."
    )


DSM_BACKEND_SOURCE = _ensure_auton_survival_available()

from auton_survival.models.dsm import DeepRecurrentSurvivalMachines
from auton_survival.models.dsm.utilities import (
    _get_padded_features,
    _get_padded_targets,
    _reshape_tensor_with_nans,
    get_optimizer,
)

DSM_BACKEND = "auton_survival"
