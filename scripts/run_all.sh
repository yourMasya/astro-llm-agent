#!/usr/bin/env bash
# Воспроизводимый запуск всего: данные -> unit-тесты -> eval (agent+baseline)
# -> стресс-тесты -> таблицы метрик.
# Требует: uv, доступ к LiteLLM-прокси (INFERENCE_API_KEY) для шага агента.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. Окружение =="
uv venv .venv --python 3.12 2>/dev/null || true
uv pip install --python .venv/bin/python -r requirements.txt

echo "== 2. Данные (идемпотентно; сгенерированные файлы коммитятся) =="
.venv/bin/python scripts/generate_dataset.py

echo "== 3. Юнит-тесты =="
.venv/bin/python -m pytest tests/ -q

echo "== 4. Eval: агент и baseline =="
.venv/bin/python eval/run_eval.py --only baseline
.venv/bin/python eval/run_eval.py --only agent

echo "== 5. Стресс-тесты =="
.venv/bin/python stress/stress_tests.py --quick

echo "== 6. Итоги =="
echo "Таблицы метрик: report/metrics_table.md"
echo "Логи вызовов:   logs/agent/*.jsonl, logs/baseline/*.jsonl"
echo "Техотчёт:       report/technical_report.md"