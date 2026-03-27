"""Always-on global color features."""

from __future__ import annotations

import numpy as np

from .constants import CONTROL_COLOR_KEYS, EPS
from .ops import safe_corr
from .views import FeatureContext


FEATURE_KEYS = CONTROL_COLOR_KEYS


def extract_color_features(ctx: FeatureContext) -> dict[str, float]:
    var_y = float(np.var(ctx.y, dtype=np.float64))
    var_cr = float(np.var(ctx.cr, dtype=np.float64))
    var_cb = float(np.var(ctx.cb, dtype=np.float64))
    return {
        "pearson_y_cr": safe_corr(ctx.y, ctx.cr),
        "pearson_y_cb": safe_corr(ctx.y, ctx.cb),
        "pearson_cr_cb": safe_corr(ctx.cr, ctx.cb),
        "energy_ratio_chroma": (var_cr + var_cb) / (var_y + EPS),
    }
