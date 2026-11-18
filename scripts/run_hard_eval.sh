#!/usr/bin/env bash
# Прогон hard-набора (протокол v2) — воспроизводит data_hard/, потолок,
# baseline и LLM-агента. Порядок зафиксирован протоколом: генерация →
# потолок → baseline → агент → финальная таблица (агент переиспользуется,
# baseline пересчитывается в logs_hard/baseline).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
ARGS=(--data-csv data_hard/dataset.csv --data-truth data_hard/ground_truth.json \
      --tasks-file tasks/tasks_hard_20.json --log-root logs_hard \
      --results-json report/metrics_hard.json \
      --results-md report/metrics_table_hard.md \
      --table-note "hard-набор: амплитуды аномалий в сигмах шума")

$PY scripts/generate_hard_dataset.py
$PY scripts/check_hard_ceiling.py
$PY eval/run_eval.py --only agent "${ARGS[@]}"
$PY eval/run_eval.py --only both "${ARGS[@]}"

echo "OK: report/metrics_hard.json, report/metrics_table_hard.md, logs_hard/"