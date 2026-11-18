#!/usr/bin/env python
"""Диагностика бенчмарка: почему F1 по аномалиям = 1.0.

Две вещи:
1) margins — фактический запас каждой инъецированной аномалии над порогом
   детектора (во сколько раз значимость точки выше порога);
2) degrade — что будет с F1 детекторов, если амплитуды инъекций сделать
   реалистичными (3-10 шумовых сигм вместо 100-1000), при тех же порогах.

Ответ на вопрос «утечка?»: ground truth (индексы) не попадает ни в промпт
агента, ни в инструменты — классической data leakage нет. Но генератор и
детектор калибровались совместно, и аномалии вставляются с запасом в сотни
сигм: задача «найти выброс» тривиально сепарабельна, поэтому F1 всегда 1.
Это вырожденность бенчмарка (design leakage), а не утечка меток.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lightagent.analysis import anomalies as an
from lightagent.config import load_config
from lightagent.data.loader import load_dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_truth():
    return json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))


def z_scores(y):
    """Числитель детектора: |y-med|/(1.4826·MAD) — те же единицы, что порог."""
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    scale = 1.4826 * mad + 1e-12
    return np.abs(y - med) / scale


def main():
    cfg = load_config()
    a = cfg.section("analysis")
    y_thr = float(a["outlier"]["threshold"])
    f_thr = float(a["flare"]["significance"])
    d_thr = float(a["dip"]["significance"])

    ds = load_dataset(cfg.root / "data" / "dataset.csv")
    curves = {cid: ds.get(cid) for cid in ds.ids()}
    truths = load_truth()

    # ---------------------------------------------------------------- margins
    out_margins, fl_margins, dip_margins = [], [], []
    n_out, n_fl, n_dip = 0, 0, 0
    for cid, c in curves.items():
        tr = truths[cid]
        y = np.asarray(c.flux)
        z = z_scores(y)
        med = np.median(y)
        # выбросы: относительно МЕДИАНЫ всей кривой (детектор так и считает),
        # но у вспышек/провалов медиана может быть смещена; для выбросов ок.
        for i in tr.get("outliers_indices", []):
            out_margins.append(z[i] / y_thr)
            n_out += 1
        # вспышки: значимость пика группы в шумовых единицах
        for (s, e2) in tr.get("flares", []):
            seg = y[s:e2 + 1]
            med_, scale = an._robust_scale(y)
            noise = np.sqrt(scale ** 2 + np.asarray(c.flux_err[s:e2 + 1]) ** 2)
            sig = (seg - med_) / np.maximum(noise, 1e-12)
            fl_margins.append(float(np.max(sig)) / f_thr)
            n_fl += 1
        # провалы: аналогично, направление вниз
        for (s, e2) in tr.get("dips", []):
            seg = y[s:e2 + 1]
            med_, scale = an._robust_scale(y)
            noise = np.sqrt(scale ** 2 + np.asarray(c.flux_err[s:e2 + 1]) ** 2)
            sig = (med_ - seg) / np.maximum(noise, 1e-12)
            dip_margins.append(float(np.max(sig)) / d_thr)
            n_dip += 1

    def stat(name, margins, n):
        arr = np.asarray(margins)
        print(f"{name}: n={n}  min={arr.min():.1f}x  "
              f"p25={np.percentile(arr,25):.1f}x  med={np.median(arr):.1f}x  "
              f"max={arr.max():.0f}x  (запас над порогом: 1x = ровно порог)")
        print(f"    доля аномалий с запасом <2x: {(arr < 2).mean()*100:.0f}%  "
              f"<1.5x: {(arr < 1.5).mean()*100:.0f}%")

    print("=== ЗАПАС АНОМАЛИЙ НАД ПОРОГОМ ДЕТЕКТОРА (текущий бенчмарк) ===")
    stat("выбросы (порог MAD {:.0f}σ)".format(y_thr), out_margins, n_out)
    stat("вспышки (порог {:.0f} шум.σ)".format(f_thr), fl_margins, n_fl)
    stat("провалы (порог {:.0f} шум.σ)".format(d_thr), dip_margins, n_dip)

    # ------------------------------------------------------- F1 при реализме
    # Перегенерируем кривые с реалистичными амплитудами инъекций:
    # выбросы 5-10σ_шума, вспышки 3-6σ, провалы 3-5σ. Пороги детектора те же.
    print("\n=== ЧТО БУДЕТ С F1 ПРИ РЕАЛИСТИЧНЫХ АМПЛИТУДАХ (те же пороги) ===")
    rng = np.random.default_rng(7)
    f1s_out, f1s_fl, f1s_dip = [], [], []
    for cid, c in curves.items():
        tr = truths[cid]
        y = np.asarray(c.flux).copy()
        dy = np.asarray(c.flux_err)
        t = np.asarray(c.t)
        sigma = np.median(dy)
        # реалистичные инъекции поверх ОБСЕРВИРУЕМОЙ кривой (те же индексы:
        # это демонстрация чувствительности, а не новый бенчмарк)
        for i in tr.get("outliers_indices", []):
            amp = rng.uniform(5.0, 10.0) * sigma * (1 if rng.random() < 0.7 else -1)
            y[i] += amp
        for (s, e2) in tr.get("flares", []):
            amp = rng.uniform(3.0, 6.0) * sigma
            y[s:e2 + 1] += amp
        for (s, e2) in tr.get("dips", []):
            amp = rng.uniform(3.0, 5.0) * sigma
            y[s:e2 + 1] -= amp

        # предсказания детекторов
        pred_out = set(an.detect_outliers(y, dy, threshold=y_thr)["indices"])
        true_out = set(tr.get("outliers_indices", []))
        if true_out or pred_out:
            tp = len(true_out & pred_out)
            fp = len(pred_out - true_out)
            fn = len(true_out - pred_out)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            f1s_out.append(f1)
        fl_pred = an.detect_flares(t, y, dy, significance=f_thr)["count"]
        fl_true = len(tr.get("flares", []))
        f1s_fl.append(min(fl_pred, fl_true) / max(fl_pred, fl_true, 1)
                      if (fl_pred or fl_true) else 1.0)
        dip_pred = an.detect_dips(t, y, dy, significance=d_thr)["count"]
        dip_true = len(tr.get("dips", []))
        f1s_dip.append(min(dip_pred, dip_true) / max(dip_pred, dip_true, 1)
                       if (dip_pred or dip_true) else 1.0)

    print(f"F1 выбросов (индексы, те же правила): "
          f"средн.={np.mean(f1s_out):.3f}, мин={min(f1s_out):.3f}")
    print(f"F1 вспышек (по числу событий):        "
          f"средн.={np.mean(f1s_fl):.3f}, мин={min(f1s_fl):.3f}")
    print(f"F1 провалов (по числу событий):       "
          f"средн.={np.mean(f1s_dip):.3f}, мин={min(f1s_dip):.3f}")
    print("\nВывод: при реалистичных амплитудах (3-10σ вместо 100-1000σ)")
    print("те же детекторы с теми же порогами перестают давать F1=1.0 —")
    print("значит текущий F1=1.0 объясняется огромным запасом инъекций,")
    print("а не качеством алгоритмов.")


if __name__ == "__main__":
    main()