from __future__ import annotations

import re


def sanitize_feature_names(cols: list[str]) -> list[str]:
    out: list[str] = []
    for c in cols:
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        if s and s[0].isdigit():
            s = "f_" + s
        out.append(s)
    return out


def ks_statistic(y_true, y_score) -> float:
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    s0 = np.sort(y_score[y_true == 0])
    s1 = np.sort(y_score[y_true == 1])
    if len(s0) == 0 or len(s1) == 0:
        return float("nan")
    scores = np.sort(np.unique(np.concatenate([s0, s1])))
    cdf0 = np.searchsorted(s0, scores, side="right") / len(s0)
    cdf1 = np.searchsorted(s1, scores, side="right") / len(s1)
    return float(np.max(np.abs(cdf1 - cdf0)))
