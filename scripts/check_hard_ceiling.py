#!/usr/bin/env python
"""Потолок достижимого на hard-наборе (протокол v2).

Для каждой задачи применяем те же детекторы с теми же порогами, что у агента
и baseline, и сравниваем с ground truth (expected). Результат — верхняя граница
качества, которую в принципе может выжать инструментальный слой: агент не
может превзойти свои инструменты. Если потолок по выбросам >= 0.95 — набор
вырожден (см. §4 протокола), это фиксируется, а не «чинится».

Метрики те же, что в eval: F1 по индексам выбросов (±2), hit-rate событий,
период ±5%, FAP.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                    # noqa: E402

from lightagent.analysis import anomalies as an                       # noqa: E402
from lightagent.config import load_config                             # noqa: E402
from lightagent.data.loader import load_dataset                       # noqa: E402
from lightagent.metrics import _set_f1, _rel_period_err               # noqa: E402
from lightagent.agent.tools import (ToolRegistry, ToolValidationError,  # noqa: E402
                                    ToolSemanticError)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    cfg = load_config()
    a = cfg.section("analysis")
    ds = load_dataset(ROOT / "data_hard" / "dataset.csv",
                      ROOT / "data_hard" / "ground_truth.json")
    tasks = json.loads((ROOT / "tasks" / "tasks_hard_20.json")
                       .read_text(encoding="utf-8"))["tasks"]
    reg = ToolRegistry(ds, cfg)

    def call(name, args):
        reg.validate_call(name, args)
        return reg.run(name, args)

    f1s, fl_hit, dip_hit, per_hit, perr = [], [], [], [], []
    periodic_correct = 0
    decided = 0
    for t in tasks:
        exp = t["expected"]
        cid = t["curve_id"]
        # выбросы: как в evaluate_task — F1 только если истина непуста
        pred_out = call("detect_outliers", {"curve_id": cid}).get("indices", [])
        true_out = exp.get("outliers_indices") or []
        if true_out:
            f1s.append(_set_f1(pred_out, true_out, tol=2))
        # вспышки/провалы (hit-rate: найдено >=1 событие)
        fl = call("detect_flares", {"curve_id": cid,
                                    "significance": a["flare"]["significance"],
                                    "min_points": a["flare"]["min_points"]})
        dp = call("detect_dips", {"curve_id": cid,
                                  "significance": a["dip"]["significance"],
                                  "min_points": a["dip"]["min_points"]})
        if exp.get("flares") is not None:
            fl_hit.append(bool(fl["count"] > 0) if exp["flares"] > 0
                          else bool(fl["count"] == 0))
        if exp.get("dips") is not None:
            dip_hit.append(bool(dp["count"] > 0) if exp["dips"] > 0
                           else bool(dp["count"] == 0))
        # период: top значимый пик + субгармоника (как baseline)
        per = call("periodogram", {"curve_id": cid, "time_unit": "days"})
        sig = [p for p in per.get("peaks", []) if p.get("significant")]
        if exp.get("periodic") is True and sig:
            cand = sig[0]
            p_days = cand["period_days"]
            if cand.get("subharmonic_similar"):
                p_days = 2.0 * cand["period_days"]
            err = _rel_period_err(p_days, exp.get("period_days"))
            perr.append(err)
            per_hit.append(err is not None and err <= 0.05)
        elif exp.get("periodic") is False:
            decided += 1
            periodic_correct += int(len(sig) == 0)

    def mean(x):
        return float(np.mean(x)) if x else None

    print("=== ПОТОЛОК ДОСТИЖИМОГО (детекторы с порогами config.yaml, "
          "hard-набор) ===")
    print(f"выбросы F1 (по задачам, где применимо): "
          f"n={len(f1s)} mean={mean(f1s):.3f} min={min(f1s) if f1s else None:.3f}")
    print(f"вспышки hit-rate: n={len(fl_hit)} mean={mean(fl_hit):.3f}")
    print(f"провалы hit-rate: n={len(dip_hit)} mean={mean(dip_hit):.3f}")
    print(f"период hit (±5%): n={len(per_hit)} mean={mean(per_hit):.3f}")
    print(f"период med rel err: {mean(perr):.4f}")
    print(f"периодичность (отрицательные задачи): "
          f"{periodic_correct}/{decided}")
    med_f1 = mean(f1s)
    if f1s and med_f1 >= 0.95:
        print("\nВНИМАНИЕ: потолок по выбросам >= 0.95 — набор всё ещё "
              "вырожден по §4 протокола v2. Параметры не подгоняются; "
              "нужен протокол v3 (меньшие амплитуды).")
    else:
        print("\nВывод: потолок < 0.95 — ось аномалий различающая; "
              "F1 ниже потолка у агента измеримы.")


if __name__ == "__main__":
    main()