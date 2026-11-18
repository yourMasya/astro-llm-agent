"""Загрузка датасета: синтетический CSV (формат id,time_days,flux,flux_err)
или любой реальный открытый CSV того же формата (OGLE/ASAS-SN экспорт).
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

from .generator import Curve


class Dataset:
    def __init__(self, curves: dict[str, Curve], truths: dict[str, dict] | None = None):
        self.curves = curves
        self.truths = truths or {}

    def get(self, curve_id: str) -> Curve:
        if curve_id not in self.curves:
            raise KeyError(f"кривая {curve_id!r} не найдена; доступны: "
                           f"{sorted(self.curves)[:10]}...")
        return self.curves[curve_id]

    def ids(self) -> list[str]:
        return sorted(self.curves)

    def __len__(self):
        return len(self.curves)


def load_dataset(csv_path, truth_path=None) -> Dataset:
    curves: dict[str, Curve] = {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows_by_id: dict[str, list] = {}
        for row in reader:
            rows_by_id.setdefault(row["id"], []).append(row)
    for cid, rows in rows_by_id.items():
        t = np.array([float(r["time_days"]) for r in rows])
        f = np.array([float(r["flux"]) for r in rows])
        e = np.array([float(r["flux_err"]) for r in rows])
        order = np.argsort(t)
        curves[cid] = Curve(id=cid, t=t[order], flux=f[order], flux_err=e[order])
    truths = {}
    if truth_path and pathlib.Path(truth_path).exists():
        with open(truth_path, "r", encoding="utf-8") as fh:
            truths = json.load(fh)
    return Dataset(curves, truths)


def default_dataset(cfg) -> Dataset:
    d = cfg.section("dataset")
    return load_dataset(cfg.root / d["out_csv"], cfg.root / d["out_truth"])