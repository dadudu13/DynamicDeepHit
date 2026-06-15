"""Compatibility imports for DSM dependencies.

Uses the local ``auton-survival`` source tree in this workspace.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTON_SURVIVAL_ROOT = REPO_ROOT / "auton-survival"
if str(AUTON_SURVIVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTON_SURVIVAL_ROOT))

from auton_survival.models.dsm import DeepRecurrentSurvivalMachines
from auton_survival.models.dsm.utilities import (
    _get_padded_features,
    _get_padded_targets,
    _reshape_tensor_with_nans,
    get_optimizer,
)

DSM_BACKEND = "auton_survival"
