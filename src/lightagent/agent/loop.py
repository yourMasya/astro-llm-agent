"""Цикл планировщика (ReAct): LLM планирует, инструменты считают.

Структура одного шага:
  LLM -> JSON {"tool": ..., "args": ...} | {"final": true, ...}
  исполнитель -> валидация, ретраи, наблюдение
  наблюдение -> обратно LLM

Guardrails (границы автономности):
- максимум шагов и вызовов инструментов на задачу;
- лимит невалидных JSON от LLM, после которого — принудительный финал;
- требование минимума вызовов инструментов (задача не «решается» текстом);
- таймаут задачи; обрезка наблюдений;
- все числа — только из инструментов: финальный ответ без единого вызова
  считается сбойным (degraded).

Каждый вызов LLM и каждый вызов инструмента пишутся в JSONL-лог.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
import uuid
from datetime import datetime, timezone

from ..config import Config
from ..data.loader import Dataset
from .executor import ToolExecutor, ToolResult
from .llm import LLMClient

SYSTEM_PROMPT = """Ты — ассистент исследователя-астронома. Твоя задача — по текстовому запросу проанализировать кривые блеска (фотометрические временные ряды) и дать проверяемый ответ.

Данные: поток в относительных единицах (среднее ≈ 1), время в СУТКАХ. Наблюдения неравномерные, есть сезонные пропуски и шум.

Тебе доступны инструменты (JSON-схемы ниже). Основные:
- get_curve_info — статистика, покрытие, пропуски;
- periodogram — периодограмма Ломба-Скаргла: топ пиков, период, FAP;
- check_alias — проверка кандидата на гармоники и суточный алиасинг;
- detect_outliers / detect_flares / detect_dips — поиск аномалий;
- plot_curve — график для отчёта.

ЖЁСТКИЕ ПРАВИЛА:
1. Все числа в ответе берутся ТОЛЬКО из результатов инструментов. Никогда не вычисляй сам и не угадывай.
2. Каждый содержательный вывод подкрепляй вызовом инструмента; минимум 2 вызова на задачу.
3. Периодичность не объявляй без periodogram: пик значим, только если fap <= fap_threshold (обычно 1e-3).
4. Единицы времени: инструмент periodogram принимает time_unit (days|hours|minutes|seconds) и возвращает период в нём. Если в запросе период просят в часах/минутах — вызывай periodogram с нужным time_unit.
5. Читай предупреждения инструментов (гармоники, суточный алиасинг, пропуски) и отражай их в notes.
6. Если инструмент вернул ошибку — прочитай текст, исправь аргументы и повтори. Не выдумывай результат и не завершай задачу по ошибке.
7. Если данных недостаточно для надёжного вывода — честно верни null в соответствующем поле и объясни в notes. Не выдавай правдоподобное за факт.
8. Не завершай работу раньше времени: собери информацию по всем частям запроса (периодичность, выбросы, вспышки, провалы, пропуски, единицы).

ФОРМАТ:
Каждый шаг, кроме финального: ровно один JSON-объект:
{"thought": "короткое обоснование", "tool": "имя_инструмента", "args": {аргументы}}

Финальный шаг — ровно один JSON-объект вида:
{"final": true, "periodic": true|false|null, "period_days": число|null, "period_unit": "days", "outliers_count": число|null, "outliers_indices": [целые индексы], "flares_count": число|null, "dips_count": число|null, "notes": "выводы, caveats, погрешности"}
- periodic: true — периодичность надёжно установлена; false — надёжно нет; null — неопределённо.
- period_days — период в СУТКАХ (всегда в сутках в финальном ответе), period_unit — всегда "days" в финале.
- outliers_indices — индексы точек-выбросов (0-based), если запрашивались.
- notes — краткие выводы: найденные аномалии, предупреждения, уверенность.

Схемы инструментов:
%s"""


class AgentSession:
    def __init__(self, task: dict, ds: Dataset, cfg: Config,
                 llm: LLMClient | None = None, executor: ToolExecutor | None = None,
                 log_dir: str | None = None, max_steps: int | None = None):
        self.task = task
        self.ds = ds
        self.cfg = cfg
        self.llm = llm or LLMClient(cfg)
        ag = cfg.section("agent")
        self.executor = executor or ToolExecutor(
            ds, cfg, obs_char_limit=ag.get("obs_char_limit", 3500),
            retries_transient=ag.get("retries_transient", 2),
            retry_backoff_s=ag.get("retry_backoff_s", [0.5, 1.0]))
        self.max_steps = max_steps if max_steps is not None \
            else int(ag.get("max_steps", 24))
        self.max_tool_calls = int(ag.get("max_tool_calls", 20))
        self.max_malformed = int(ag.get("max_malformed", 3))
        self.max_consecutive_errors = int(ag.get("max_consecutive_errors", 3))
        self.timeout_s = float(ag.get("timeout_s", 240))
        self.require_min_tool_calls = int(ag.get("require_min_tool_calls", 2))
        self.log_dir = pathlib.Path(log_dir) if log_dir else \
            cfg.root / "logs" / "agent"
        self.log_path = self.log_dir / f"{task['id']}.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "task_id": task["id"], "steps": 0, "tool_calls": 0,
            "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "est_cost_usd": 0.0, "retries_total": 0,
            "validation_errors": 0, "semantic_errors": 0, "tool_errors": 0,
            "malformed_json": 0, "duration_s": 0.0, "degraded": False,
            "degraded_reason": None, "tool_names": [],
        }

    # -- логи -------------------------------------------------------------

    def _log(self, **data):
        rec = {"ts": datetime.now(timezone.utc).isoformat(),
               "task_id": self.task["id"], **data}
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # -- парсинг JSON от LLM ----------------------------------------------

    @staticmethod
    def extract_json(text: str):
        t = text.strip()
        if t.startswith("```"):
            m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
            if m:
                t = m.group(1).strip()
        try:
            return json.loads(t), None
        except json.JSONDecodeError:
            pass
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(t[start:end + 1]), None
            except json.JSONDecodeError as exc:
                return None, f"невалидный JSON: {exc}"
        return None, "JSON-объект не найден в ответе"

    # -- один шаг ----------------------------------------------------------

    def _is_final(self, action: dict) -> bool:
        return isinstance(action, dict) and action.get("final") is True

    def _step(self, messages) -> tuple[str, float]:
        """Один вызов LLM. Возвращает (текст, обновлённый cost)."""
        t0 = time.monotonic()
        text, meta = self.llm.chat(messages)
        dur = time.monotonic() - t0
        self.stats["steps"] += 1
        self.stats["llm_calls"] += 1
        self.stats["prompt_tokens"] += meta.get("prompt_tokens", 0)
        self.stats["completion_tokens"] += meta.get("completion_tokens", 0)
        cost = self.llm.estimate_cost(meta.get("prompt_tokens", 0),
                                      meta.get("completion_tokens", 0))
        self.stats["est_cost_usd"] += cost
        self._log(kind="llm", step=self.stats["steps"], text=text[:1500],
                  prompt_tokens=meta.get("prompt_tokens", 0),
                  completion_tokens=meta.get("completion_tokens", 0),
                  cost_usd=round(cost, 6), duration_s=round(dur, 2))
        return text

    def _tool_call(self, action: dict) -> tuple[ToolResult, str]:
        t0 = time.monotonic()
        res = self.executor.call(action["tool"], action.get("args", {}))
        dur = time.monotonic() - t0
        self.stats["tool_calls"] += 1
        self.stats["retries_total"] += res.retries
        if res.status == "validation_error":
            self.stats["validation_errors"] += 1
        elif res.status == "semantic_error":
            self.stats["semantic_errors"] += 1
        elif res.status == "error":
            self.stats["tool_errors"] += 1
        if res.name not in self.stats["tool_names"]:
            self.stats["tool_names"].append(res.name)
        self._log(kind="tool", step=self.stats["steps"], tool=res.name,
                  args=action.get("args", {}), status=res.status,
                  error=res.error, retries=res.retries,
                  duration_ms=round(res.duration_ms, 2),
                  text=res.text[:800])
        obs = self.executor.observation_text(res)
        return res, obs

    # -- главный цикл -------------------------------------------------------

    def run(self) -> dict:
        t_start = time.monotonic()
        ag = self.cfg.section("agent")
        sys_prompt = SYSTEM_PROMPT % self.executor.registry.spec_text()
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": self.task["query"]},
        ]
        final = None
        malformed_count = 0
        min_calls_nag = 0
        tool_error_streak: dict[str, int] = {}
        degraded = False
        reason = None

        while self.stats["steps"] < self.max_steps:
            if time.monotonic() - t_start > self.timeout_s:
                degraded, reason = True, f"таймаут {self.timeout_s:.0f} с"
                break

            text = self._step(messages)
            action, err = self.extract_json(text)
            if action is None:
                malformed_count += 1
                self.stats["malformed_json"] += 1
                if malformed_count > self.max_malformed:
                    degraded, reason = True, "планировщик не формирует валидный JSON"
                    break
                messages.append({"role": "assistant", "content": text[:1500]})
                messages.append({"role": "user", "content":
                    f"Ошибка формата ({err}). Верни РОВНО один JSON-объект без "
                    f"лишнего текста: {{\"thought\": ..., \"tool\": ..., \"args\": ...}} "
                    f"или финал {{\"final\": true, ...}}."})
                continue

            if self._is_final(action):
                if self.stats["tool_calls"] < self.require_min_tool_calls and \
                        min_calls_nag < 1:
                    # не даём «решить» задачу без инструментов
                    min_calls_nag += 1
                    messages.append({"role": "assistant", "content": text[:1500]})
                    messages.append({"role": "user", "content":
                        "Ты завершил задачу, не вызвав ни одного инструмента "
                        f"(вызовов: {self.stats['tool_calls']}, нужно минимум "
                        f"{self.require_min_tool_calls}). Все числа должны "
                        "приходить из инструментов. Продолжай: вызови "
                        "инструменты и только потом верни финал."})
                    continue
                final = action
                break

            if "tool" not in action or "args" not in action:
                malformed_count += 1
                self.stats["malformed_json"] += 1
                if malformed_count > self.max_malformed:
                    degraded, reason = True, "планировщик не формирует вызов"
                    break
                messages.append({"role": "assistant", "content": text[:1500]})
                messages.append({"role": "user", "content":
                    "Объект должен содержать поля \"tool\" и \"args\" (или "
                    "\"final\": true). Исправь."})
                continue

            if self.stats["tool_calls"] >= self.max_tool_calls:
                degraded, reason = True, f"исчерпан лимит вызовов ({self.max_tool_calls})"
                break

            res, obs = self._tool_call(action)
            # Guardrail: инструмент стабильно недоступен (N сбоев подряд одного
            # инструмента) — агент не тратит весь бюджет на повторные попытки,
            # а честно завершает с сообщением об ошибке.
            if res.status == "error":
                tool_error_streak[res.name] = tool_error_streak.get(res.name, 0) + 1
            else:
                tool_error_streak.pop(res.name, None)
            if tool_error_streak.get(res.name, 0) >= self.max_consecutive_errors:
                degraded, reason = True, (
                    f"инструмент {res.name} недоступен "
                    f"({self.max_consecutive_errors} сбоев подряд)")
                break
            messages.append({"role": "assistant", "content": text[:1500]})
            messages.append({"role": "user", "content": obs})

        if final is None and reason is None:
            degraded, reason = True, f"исчерпан лимит шагов ({self.max_steps})"

        if final is None:
            final = {
                "final": True, "periodic": None, "period_days": None,
                "period_unit": "days", "outliers_count": None,
                "outliers_indices": [], "flares_count": None,
                "dips_count": None,
                "notes": f"задача не завершена: {reason}",
            }

        self.stats["degraded"] = degraded
        self.stats["degraded_reason"] = reason
        self.stats["duration_s"] = round(time.monotonic() - t_start, 2)
        self.stats["final"] = final
        self._log(kind="final", final=final, stats=self.stats)
        return {"final": final, "stats": self.stats}