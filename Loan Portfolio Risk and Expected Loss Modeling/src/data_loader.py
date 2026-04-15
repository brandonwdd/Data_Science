from __future__ import annotations

import config
from src.preprocessing import load_accepted


def load_training_frame(nrows: int | None = None):
    path = config.resolve_accepted_csv()
    return load_accepted(path, nrows=nrows)
