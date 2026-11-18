#!/usr/bin/env python
"""Генерация бенчмарк-датасета кривых блеска + ground truth + карточки.

Детерминированно (seed и id кривых задают ГПСЧ), результат коммитится:
data/dataset.csv, data/ground_truth.json, data/dataset_card.md.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lightagent.config import load_config          # noqa: E402
from lightagent.data.generator import generate_dataset, save_dataset  # noqa: E402


def main():
    cfg = load_config()
    d = cfg.section("dataset")
    curves, truths = generate_dataset(
        d["n_curves"], seed=d["seed"],
        seasons=d["seasons"], season_len_d=d["season_len_d"],
        gap_d=d["gap_d"], cadence_base_d=d["cadence_base_d"],
        cadence_jitter=d["cadence_jitter"],
        drop_fraction=d["drop_fraction"], noise_base=d["noise_base"])
    save_dataset(curves, truths,
                 cfg.root / d["out_csv"], cfg.root / d["out_truth"],
                 cfg.root / d["out_card"])
    print(f"OK: {len(curves)} кривых -> {d['out_csv']}, {d['out_truth']}")


if __name__ == "__main__":
    main()