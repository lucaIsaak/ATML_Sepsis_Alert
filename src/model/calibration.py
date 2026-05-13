"""
Isotonic calibration wrapper for HistGradientBoostingClassifier.

Defined in a standalone module so that joblib/pickle can always resolve
the class regardless of how the process is started (uvicorn worker,
direct script, subprocess).  If this were defined inside train.py, the
class would be pickled as ``src.model.train._IsotonicCalibrated``; when
uvicorn loads train.py as ``__mp_main__`` the lookup fails.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrated:
    """
    Pre-fitted HistGBM + isotonic calibration layer.

    Replaces ``CalibratedClassifierCV(model, cv='prefit', method='isotonic')``,
    which was removed in sklearn 1.4.  Semantics are identical: the base
    model is never refitted; only the isotonic mapping is learned on the
    held-out calibration split.

    Exposes ``predict_proba`` so it is a drop-in for CalibratedClassifierCV
    everywhere in the codebase (inference, evaluate.py, SHAP wrapper).
    """

    def __init__(
        self,
        base_model: HistGradientBoostingClassifier,
        calibrator: IsotonicRegression,
    ) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:  # noqa: ANN001
        raw = self.base_model.predict_proba(X)[:, 1]
        # Clip to (0, 0.999) — isotonic regression extrapolates to exactly 1.0
        # for inputs above the training range; a calibrated model should never
        # claim absolute certainty.
        cal = np.clip(self.calibrator.predict(raw), 0.001, 0.999)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X) -> np.ndarray:  # noqa: ANN001
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
