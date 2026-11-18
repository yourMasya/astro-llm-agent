#!/usr/bin/env python
"""Прогон eval: агент и детерминированный baseline на 20 задачах.

Запуск:
  python eval/run_eval.py                     # baseline + агент
  python eval/run_eval.py --only baseline     # только baseline (без LLM)
  python eval/run_eval.py --only agent
  python eval/run_eval.py --limit 3           # первые 3 задачи (отладка)
  python eval/run_eval.py --tasks t01,t06

Результаты:
- logs/agent/<task>.jsonl и logs/baseline/<task>.jsonl — вызовы инструментов;
- report/metrics.json — сырые метрики;
- report/metrics_table.md — таблица метрик (сводка + по задачам).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lightagent.config import load_config        # noqa: E402
from lightagent.data.loader import load_dataset  # noqa: E402
from lightagent.agent.loop import AgentSession   # noqa: E402
from lightagent.agent.llm import LLMClient       # noqa: E402
from lightagent.baseline import BaselineRunner   # noqa: E402
from lightagent.metrics import (evaluate_task, aggregate, fmt_pct)  # noqa: E402


def load_tasks(cfg, only_tasks=None, limit=None, tasks_file=None):
    if tasks_file is None:
        tasks_file = cfg.root / cfg.get("eval", "tasks_file",
                                        default="tasks/tasks_20.json")
    data = json.loads(pathlib.Path(tasks_file).read_text(encoding="utf-8"))
    tasks = data["tasks"]
    if only_tasks:
        tasks = [t for t in tasks if t["id"] in only_tasks]
    if limit:
        tasks = tasks[:limit]
    return tasks


def run_one(runner_kind, task, ds, cfg, max_attempts=3, sleep_between=25.0,
            log_root=None):
    import time as _time
    if runner_kind == "baseline":
        runner = BaselineRunner(ds, cfg,
                                log_dir=(log_root / "baseline") if log_root
                                else None)
        return runner.run(task)
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            session = AgentSession(task, ds, cfg, llm=LLMClient(cfg),
                                   log_dir=(log_root / "agent") if log_root
                                   else None)
            return session.run()
        except Exception as exc:  # LLMError, сеть, rate-limit
            last_err = exc
            print(f"    попытка {attempt}/{max_attempts} не удалась: {exc}", flush=True)
            if attempt < max_attempts:
                _time.sleep(sleep_between * attempt)
    # устойчивый отказ: честный degraded-ответ
    return {"final": {
        "final": True, "periodic": None, "period_days": None,
        "period_unit": "days", "outliers_count": None, "outliers_indices": [],
        "flares_count": None, "dips_count": None,
        "notes": f"LLM недоступен после {max_attempts} попыток: {last_err}"},
        "stats": {"task_id": task["id"], "tool_calls": 0, "llm_calls": 0,
                  "prompt_tokens": 0, "completion_tokens": 0, "est_cost_usd": 0.0,
                  "retries_total": 0, "validation_errors": 0,
                  "semantic_errors": 0, "tool_errors": 0, "malformed_json": 0,
                  "degraded": True, "degraded_reason": str(last_err),
                  "duration_s": 0.0, "final": None, "tool_names": []}}


def load_reused(kind, task_ids, path):
    """Взять готовые метрики runner-а из прошлого прогона, если покрывают всё."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["task"]["id"]: r.get(kind) for r in data.get("rows", [])
            if r.get(kind) is not None}
    if all(tid in rows for tid in task_ids):
        return rows
    return {}


def load_final_from_log(path):
    """Восстановить результат задачи из лога (kind=final)."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "final":
            return {"final": rec.get("final", {}), "stats": rec.get("stats", {})}
    return None


def reuse_or_run(kind, task, ds, cfg, reused):
    if task["id"] in reused:
        return reused[task["id"]]
    res = run_one(kind, task, ds, cfg)
    return evaluate_task(
        task, res, task["expected"],
        period_tolerance_frac=cfg.get("eval", "period_tolerance_frac", default=0.05),
        log_path=cfg.root / "logs" / kind / f"{task['id']}.jsonl")


def fmt_metric(val):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _fmt_none(val):
    return "—" if val is None else str(val)


def build_md(rows, agent_agg, base_agg, tasks, *, data_note=""):
    lines = ["# Метрики: LLM-агент vs детерминированный baseline", ""]
    lines.append(f"Задач: **{len(tasks)}**. "
                 f"Данные: {data_note or '`data/dataset.csv`'}. "
                 "Стоимость LLM — оценка по тарифам deepseek-chat "
                 "(config.yaml).")
    lines.append("")
    lines.append("## Сводка")
    lines.append("")
    lines.append("| Метрика | Агент | Baseline |")
    lines.append("|---|---|---|")
    metric_names = [
        ("periodic_acc", "Точность решения о периодичности (по решённым)"),
        ("periodic_decided_frac", "Доля решённых задач о периодичности"),
        ("period_hit_rate", "Доля задач с верным периодом (±5%)"),
        ("median_period_rel_err", "Медианная относит. ошибка периода"),
        ("outliers_f1_mean", "F1 по выбросам (допуск ±2 индекса)"),
        ("flares_hit_rate", "Детекция вспышек (hit rate)"),
        ("dips_hit_rate", "Детекция провалов (hit rate)"),
        ("caveat_hit_rate", "Отражение обязательных оговорок"),
        ("unit_handled_rate", "Корректная единица времени (по логам)"),
        ("success_rate", "Доля успешно завершённых задач"),
        ("avg_tool_calls", "Ср. число вызовов инструментов"),
        ("avg_retries", "Ср. число повторных попыток"),
        ("validation_errors_total", "Ошибки валидации аргументов (всего)"),
        ("semantic_errors_total", "Семантические ошибки инструментов (всего)"),
        ("tool_errors_total", "Ошибки инструментов (всего)"),
        ("malformed_json_total", "Невалидный JSON планировщика (всего)"),
        ("total_tokens", "Токенов (вход+выход)"),
        ("total_cost_usd", "Оценка стоимости, USD"),
        ("avg_duration_s", "Ср. время на задачу, с"),
    ]
    for key, name in metric_names:
        lines.append(f"| {name} | {fmt_metric(agent_agg.get(key))} | "
                     f"{fmt_metric(base_agg.get(key))} |")
    lines.append("")
    lines.append("## По задачам")
    lines.append("")
    lines.append("T — периодичность true, F — false, ? — запрос её не касался; "
                 "— — метрика неприменима; стоимость — только у агента (LLM).")
    lines.append("")
    lines.append("| Задача | Кривая | periodic агент/base | период агент/base | "
                 "выбросы F1 агент/base | вспышки агент/base | провалы агент/base | "
                 "успех агент/base | вызовы | стоимость, $ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        t = r["task"]
        a, b = r.get("agent"), r.get("baseline")
        def pa(x):
            v = x.get("periodic_pred") if x else None
            return {True: "T", False: "F", None: "?"}[v]
        def per(x):
            v = (x.get("period_rel_err") if x else None)
            return "—" if v is None else f"{v:.3f}"
        def f1(x):
            v = x.get("outliers_f1") if x else None
            return "—" if v is None else f"{v:.2f}"
        def hit(x, k):
            v = x.get(k) if x else None
            return "—" if v is None else ("+" if v else "–")
        def ok(x):
            return "+" if (x and x.get("success")) else "–"
        lines.append(
            f"| {t['id']} | {t['curve_id']} | {pa(a)}/{pa(b)} | "
            f"{per(a)}/{per(b)} | {f1(a)}/{f1(b)} | "
            f"{hit(a,'flares_hit')}/{hit(b,'flares_hit')} | "
            f"{hit(a,'dips_hit')}/{hit(b,'dips_hit')} | "
            f"{ok(a)}/{ok(b)} | {(a or {}).get('tool_calls','–')}/{(b or {}).get('tool_calls','–')} | "
            f"{(a or {}).get('est_cost_usd', 0):.4f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["agent", "baseline", "both"], default="both")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tasks", type=str, default="",
                    help="список id задач через запятую")
    ap.add_argument("--no-reuse", action="store_true",
                    help="не переиспользовать метрики прошлых прогонов")
    ap.add_argument("--reuse-logs", action="store_true",
                    help="для агента брать финальные ответы из logs/*.jsonl "
                         "вместо новых вызовов LLM (пересчёт метрик без затрат)")
    ap.add_argument("--data-csv", default=None,
                    help="CSV с кривыми (по умолчанию data/dataset.csv)")
    ap.add_argument("--data-truth", default=None,
                    help="JSON с ground truth (по умолчанию data/ground_truth.json)")
    ap.add_argument("--tasks-file", default=None,
                    help="файл задач (по умолчанию tasks/tasks_20.json)")
    ap.add_argument("--log-root", default=None,
                    help="каталог логов вызовов (по умолчанию logs/)")
    ap.add_argument("--results-json", default=None,
                    help="куда писать метрики (по умолчанию report/metrics.json)")
    ap.add_argument("--results-md", default=None,
                    help="куда писать таблицу (по умолчанию report/metrics_table.md)")
    ap.add_argument("--table-note", default="",
                    help="описание набора данных для шапки таблицы")
    args = ap.parse_args()

    cfg = load_config()
    d = cfg.section("dataset")
    data_csv = pathlib.Path(args.data_csv) if args.data_csv \
        else cfg.root / d["out_csv"]
    data_truth = pathlib.Path(args.data_truth) if args.data_truth \
        else cfg.root / d["out_truth"]
    tasks_file = pathlib.Path(args.tasks_file) if args.tasks_file \
        else cfg.root / cfg.get("eval", "tasks_file", default="tasks/tasks_20.json")
    log_root = pathlib.Path(args.log_root) if args.log_root \
        else cfg.root / cfg.get("eval", "logs_dir", default="logs")
    results_path = pathlib.Path(args.results_json) if args.results_json \
        else cfg.root / "report" / "metrics.json"
    results_md_path = pathlib.Path(args.results_md) if args.results_md \
        else cfg.root / "report" / "metrics_table.md"

    ds = load_dataset(data_csv, data_truth)
    tasks = load_tasks(cfg, tasks_file=tasks_file,
                       only_tasks=args.tasks.split(",") if args.tasks else None,
                       limit=args.limit)
    reused = {}
    if not args.no_reuse:
        for kind in ("agent", "baseline"):
            if args.only in (kind, "both"):
                part = load_reused(kind, [t["id"] for t in tasks], results_path)
                if part:
                    reused.update({(kind, tid): v for tid, v in part.items()})
        if reused:
            print(f"Переиспользованы результаты прошлых прогонов: "
                  f"{sum(1 for (k, _) in reused if k == 'agent')} agent, "
                  f"{sum(1 for (k, _) in reused if k == 'baseline')} baseline")
    print(f"Задач: {len(tasks)}; агент={args.only in ('agent','both')}, "
          f"baseline={args.only in ('baseline','both')}")

    rows = []
    for i, task in enumerate(tasks, 1):
        row = {"task": task}
        print(f"[{i}/{len(tasks)}] {task['id']} ({task['curve_id']})...", flush=True)
        if args.only in ("agent", "both"):
            if ("agent", task["id"]) in reused:
                row["agent"] = reused[("agent", task["id"])]
            else:
                res = None
                if args.reuse_logs:
                    res = load_final_from_log(log_root / "agent"
                                              / f"{task['id']}.jsonl")
                if res is None:
                    res = run_one("agent", task, ds, cfg, log_root=log_root)
                row["agent"] = evaluate_task(
                    task, res, task["expected"],
                    period_tolerance_frac=cfg.get("eval", "period_tolerance_frac", default=0.05),
                    log_path=log_root / "agent" / f"{task['id']}.jsonl")
        if args.only in ("baseline", "both"):
            if ("baseline", task["id"]) in reused:
                row["baseline"] = reused[("baseline", task["id"])]
            else:
                res = run_one("baseline", task, ds, cfg, log_root=log_root)
                row["baseline"] = evaluate_task(
                    task, res, task["expected"],
                    period_tolerance_frac=cfg.get("eval", "period_tolerance_frac", default=0.05),
                    log_path=log_root / "baseline" / f"{task['id']}.jsonl")
        rows.append(row)
        if row.get("agent"):
            print(f"  агент: успех={row['agent']['success']}, "
                  f"вызовы={row['agent']['tool_calls']}, "
                  f"стоимость=${row['agent']['est_cost_usd']:.5f}")
        if row.get("baseline"):
            print(f"  baseline: успех={row['baseline']['success']}, "
                  f"вызовы={row['baseline']['tool_calls']}")

    out = {"rows": rows}
    if rows and rows[0].get("agent"):
        out["agent_agg"] = aggregate([r["agent"] for r in rows])
    if rows and rows[0].get("baseline"):
        out["baseline_agg"] = aggregate([r["baseline"] for r in rows])

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
    md = build_md(rows, out.get("agent_agg", {}), out.get("baseline_agg", {}),
                  tasks, data_note=args.table_note)
    results_md_path.write_text(md, encoding="utf-8")
    print(f"Записано: {results_path}, {results_md_path}")

    agg = out.get("agent_agg", {})
    bagg = out.get("baseline_agg", {})
    print("\n=== Сводка ===")
    for k, name in [("periodic_acc", "Точность периодичности"),
                    ("period_hit_rate", "Точность периода"),
                    ("outliers_f1_mean", "F1 выбросы"),
                    ("success_rate", "Успешность"),
                    ("total_cost_usd", "Стоимость, $")]:
        print(f"{name:<24} агент={fmt_metric(agg.get(k))}  "
              f"baseline={fmt_metric(bagg.get(k))}")


if __name__ == "__main__":
    main()