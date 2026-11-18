#!/usr/bin/env python
"""Генерация hard-набора по протоколу v2 (см. report/measurement_protocol_v2.md).

Создаёт 20 кривых h000-h019: те же типы сигналов, что в v1-задачах
(по data/ground_truth.json исходных bXXX), те же параметры наземного обзора,
но амплитуды аномалий — реалистичные множители к шуму ряда (σ), а не
абсолютные единицы. Детекторы, их пороги и метрики НЕ меняются.

Результаты:
- data_hard/dataset.csv, data_hard/ground_truth.json, data_hard/dataset_card.md
- tasks/tasks_hard_20.json — копия задач v1 с curve_id hXXX и ground truth,
  пересчитанным из truth новых кривых.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lightagent.config import load_config              # noqa: E402
from lightagent.data.generator import generate_curve, save_dataset  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Зафиксированные протоколом v2 распределения (см. §2 протокола).
OUTLIER_AMP_SIGMA = (3.0, 8.0)
FLARE_AMP_SIGMA = (4.0, 10.0)
DIP_DEPTH_SIGMA = (3.0, 8.0)


def obs_kw_from_cfg(cfg) -> dict:
    d = cfg.section("dataset")
    return {k: d[k] for k in (
        "seasons", "season_len_d", "gap_d", "cadence_base_d",
        "cadence_jitter", "drop_fraction", "noise_base")}


def main():
    cfg = load_config()
    tasks_v1 = json.loads((ROOT / "tasks" / "tasks_20.json")
                          .read_text(encoding="utf-8"))["tasks"]
    truths_v1 = json.loads((ROOT / "data" / "ground_truth.json")
                           .read_text(encoding="utf-8"))

    obs = obs_kw_from_cfg(cfg)
    curves, truths = {}, {}
    tasks_hard = []
    for i, task in enumerate(tasks_v1):
        src_id = task["curve_id"]
        ctype = truths_v1[src_id]["type"]
        cid = f"h{i:03d}"
        curve, truth = generate_curve(
            cid, ctype,
            outlier_amp_sigma=OUTLIER_AMP_SIGMA,
            flare_amp_sigma=FLARE_AMP_SIGMA,
            dip_depth_sigma=DIP_DEPTH_SIGMA,
            **obs)
        curves[cid] = curve
        truths[cid] = truth

        # копия задачи: тот же запрос/структура, новая кривая и truth.
        # ВАЖНО: заменить старый id (bXXX) и в тексте запроса тоже — иначе
        # агент честно ищет несуществующую кривую (поймано на первом прогоне).
        t2 = copy.deepcopy(task)
        src_id_re = re.compile(rf"\b{re.escape(src_id)}\b")
        t2["curve_id"] = cid
        t2["query"] = src_id_re.sub(cid, t2["query"])
        exp = t2["expected"]
        if exp.get("periodic") is True:
            exp["period_days"] = truth["P_true"]
        if exp.get("outliers_indices") is not None:
            exp["outliers_indices"] = truth["outliers_indices"]
        if exp.get("flares") is not None:
            exp["flares"] = len(truth["flares"])
        if exp.get("dips") is not None:
            exp["dips"] = len(truth["dips"])
        tasks_hard.append(t2)
        print(f"{src_id}({ctype}) -> {cid}: P={truth['P_true']}, "
              f"out={len(truth['outliers_indices'])}, "
              f"fl={len(truth['flares'])}, dips={len(truth['dips'])}")

    out_dir = ROOT / "data_hard"
    save_dataset(curves, truths, out_dir / "dataset.csv",
                 out_dir / "ground_truth.json", out_dir / "dataset_card.md")

    out_tasks = ROOT / "tasks" / "tasks_hard_20.json"
    payload = {
        "description":
        "20 задач hard-набора (протокол v2): те же запросы, что v1, "
        "кривые h000-h019 с реалистичными амплитудами аномалий (в σ шума). "
        "Ground truth пересчитан из генератора для новых кривых.",
        "tasks": tasks_hard,
    }
    with open(out_tasks, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"OK: {len(curves)} кривых -> data_hard/, {out_tasks.name}")


if __name__ == "__main__":
    main()