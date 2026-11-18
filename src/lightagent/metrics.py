"""Количественные метрики сравнения агента и baseline.

Точность аналитических выводов:
- periodic_acc      — точность бинарного решения о периодичности (по решённым);
- period_hit        — период найден с относительной погрешностью <= tol;
- period_rel_err    — |P_pred - P_true| / P_true;
- unit_handled      — инструмент вызывался с единицей из запроса (по логам);
- outliers_f1       — F1 по множеству индексов выбросов (допуск ±N позиций);
- flares_hit/dips_hit — обнаружено >=1 событие при их наличии в истине;
- caveat_hit        — в ответе отражён обязательный caveat (алиасинг/пропуски).

Стоимость: токены, оценка USD (тарифы deepseek-chat), вызовы инструментов,
длительность.

Надёжность: успешность задачи (не degraded), повторные попытки, ошибки
валидации/семантики/инструментов, число невалидных JSON от планировщика.
"""

from __future__ import annotations

import math

from .agent.loop import AgentSession


def _set_f1(pred_idx, true_idx, tol=2):
    """F1 по индексам с окрестностным допуском."""
    pred = set(int(i) for i in pred_idx)
    true = set(int(i) for i in true_idx)
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    hit = set()
    for i in pred:
        if any(abs(i - j) <= tol for j in true):
            hit.add(i)
    tp = len(hit)
    fp = len(pred) - tp
    # истинные, покрытые допуском любого предсказанного
    tp_true = sum(1 for j in true if any(abs(i - j) <= tol for i in pred))
    fn = len(true) - tp_true
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp_true / (tp_true + fn) if (tp_true + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _rel_period_err(pred_days, true_days):
    if pred_days is None or not true_days:
        return None
    return abs(pred_days - true_days) / true_days


def _unit_handled(task, result, log_path=None):
    """Единица запроса обработана: инструмент вызывался с нужным time_unit."""
    exp_unit = (task.get("expected") or {}).get("period_unit", "days")
    if exp_unit == "days":
        return None  # не требуется
    return _scan_logs_for_unit(log_path, exp_unit)


def _scan_logs_for_unit(log_path, unit):
    import json as _json
    if not log_path or not log_path.exists():
        return False
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if rec.get("kind") == "tool" and rec.get("tool") == "periodogram":
                args = rec.get("args") or {}
                if args.get("time_unit") == unit:
                    return True
    except OSError:
        return False
    return False


def evaluate_task(task: dict, result: dict, expected: dict, *,
                  period_tolerance_frac=0.05, index_neighbor_tol=2,
                  log_path=None, caveat_keywords=("алиас", "гармоник",
                                                  "пропуск", "покрыт",
                                                  "неоднознач", "низк")):
    final = result.get("final", {})
    notes = (final.get("notes") or "").lower()
    out = {}

    # -- периодичность
    exp_per = expected.get("periodic")
    pred_per = final.get("periodic")
    out["periodic_expected"] = exp_per
    out["periodic_pred"] = pred_per
    if isinstance(exp_per, bool) and isinstance(pred_per, bool):
        out["periodic_correct"] = bool(exp_per == pred_per)
        out["periodic_decided"] = True
    else:
        out["periodic_correct"] = False
        out["periodic_decided"] = False

    # -- период
    true_p = expected.get("period_days")
    pred_p = final.get("period_days")
    unit_hint = expected.get("period_unit", "days")
    err = _rel_period_err(pred_p, true_p) if (true_p and exp_per) else None
    if true_p and exp_per:
        out["period_rel_err"] = err
        out["period_hit"] = bool(err is not None and err <= period_tolerance_frac)
    else:
        out["period_rel_err"] = None
        out["period_hit"] = None   # не применимо: истина непериодична
    out["unit_handled"] = _unit_handled(task, result, log_path) \
        if unit_hint != "days" else None

    # -- аномалии
    true_out = expected.get("outliers_indices", [])
    pred_out = final.get("outliers_indices") or []
    n_true_out = len(true_out)
    n_pred_out = len(pred_out)
    if n_true_out == 0:
        # F1 не оценивается, если задача не про выбросы: пики вспышек могут
        # честно ловиться MAD-методом как выбросы, это не ошибка.
        out["outliers_f1"] = None
    else:
        out["outliers_f1"] = _set_f1(pred_out, true_out, index_neighbor_tol)
    out["outliers_count_err"] = abs(n_pred_out - n_true_out)

    for key, pred_key, exp_key in (
            ("flares_hit", "flares_count", "flares"),
            ("dips_hit", "dips_count", "dips")):
        exp_n = expected.get(exp_key)
        pred_n = final.get(pred_key)
        if exp_n is None:
            out[key] = None
        else:
            out[key] = bool(pred_n is not None and pred_n > 0) if exp_n > 0 \
                else bool(pred_n is not None and pred_n == 0)

    # -- caveat
    if expected.get("caveat_required"):
        out["caveat_hit"] = any(k in notes for k in caveat_keywords)
    else:
        out["caveat_hit"] = None

    # -- надёжность
    stats = result.get("stats", {})
    out["success"] = not stats.get("degraded", False)
    out["degraded_reason"] = stats.get("degraded_reason")
    out["tool_calls"] = stats.get("tool_calls", 0)
    out["retries"] = stats.get("retries_total", 0)
    out["validation_errors"] = stats.get("validation_errors", 0)
    out["semantic_errors"] = stats.get("semantic_errors", 0)
    out["tool_errors"] = stats.get("tool_errors", 0)
    out["malformed_json"] = stats.get("malformed_json", 0)
    out["est_cost_usd"] = stats.get("est_cost_usd", 0.0)
    out["duration_s"] = stats.get("duration_s", 0.0)
    out["tokens"] = stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0)
    return out


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    def count(key):
        return sum(1 for r in rows if r.get(key))
    decided = [r for r in rows if r.get("periodic_decided")]
    perr = [r for r in rows if r.get("period_rel_err") is not None]
    return {
        "n_tasks": n,
        "periodic_acc": (sum(r["periodic_correct"] for r in decided) / len(decided)
                         if decided else None),
        "periodic_decided_frac": len(decided) / n if n else None,
        "period_hit_rate": mean("period_hit"),
        "median_period_rel_err": (sorted(r["period_rel_err"]
                                         for r in perr)[len(perr) // 2]
                                  if perr else None),
        "outliers_f1_mean": mean("outliers_f1"),
        "flares_hit_rate": mean("flares_hit"),
        "dips_hit_rate": mean("dips_hit"),
        "caveat_hit_rate": mean("caveat_hit"),
        "unit_handled_rate": mean("unit_handled"),
        "success_rate": count("success") / n,
        "avg_tool_calls": mean("tool_calls"),
        "avg_retries": mean("retries"),
        "validation_errors_total": sum(r.get("validation_errors", 0) for r in rows),
        "semantic_errors_total": sum(r.get("semantic_errors", 0) for r in rows),
        "tool_errors_total": sum(r.get("tool_errors", 0) for r in rows),
        "malformed_json_total": sum(r.get("malformed_json", 0) for r in rows),
        "total_cost_usd": round(sum(r.get("est_cost_usd", 0.0) for r in rows), 6),
        "total_tokens": sum(r.get("tokens", 0) for r in rows),
        "avg_duration_s": mean("duration_s"),
    }


def fmt_pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def metric_value(m):
    """Численное значение метрики для сортировки (None -> -inf)."""
    if m is None:
        return float("-inf")
    return float(m)