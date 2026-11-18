"""Юнит-тесты ReAct-цикла: guardrails на лимиты и устойчивые отказы.

LLM здесь — заглушка: детерминированные последовательности действий.
Проверяется поведение guardrail-ов, а не качество рассуждений модели.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from lightagent.config import load_config
from lightagent.data.loader import load_dataset
from lightagent.agent.executor import ToolExecutor
from lightagent.agent.loop import AgentSession

META = {"prompt_tokens": 10, "completion_tokens": 10}


@pytest.fixture(scope="module")
def ctx():
    cfg = load_config()
    d = cfg.section("dataset")
    ds = load_dataset(cfg.root / d["out_csv"], cfg.root / d["out_truth"])
    return cfg, ds


class FakeLLM:
    """Возвращает действия из списка по очереди; последнее повторяет."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        action = self.actions[min(self.calls - 1, len(self.actions) - 1)]
        return json.dumps(action, ensure_ascii=False), dict(META)

    @staticmethod
    def estimate_cost(prompt_tokens, completion_tokens):
        return 0.0


def _outliers_action():
    return {"thought": "x", "tool": "detect_outliers", "args": {"curve_id": "b001"}}


def test_loop_stops_on_persistent_tool_failure(ctx, tmp_path):
    """Стабильно падающий инструмент: агент завершается по guardrail
    (N сбоев подряд), а не жжёт весь бюджет шагов."""
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg)
    ex.inject_failure("detect_outliers", lambda: RuntimeError("backend down"),
                      attempts=999)
    llm = FakeLLM([_outliers_action()])  # агент упрямо зовёт один инструмент
    sess = AgentSession({"id": "t_stress", "curve_id": "b001",
                         "query": "Сколько выбросов в b001?"},
                        ds, cfg, llm=llm, executor=ex, max_steps=24,
                        log_dir=str(tmp_path))
    out = sess.run()
    assert out["stats"]["degraded"] is True
    assert out["stats"]["steps"] < 10  # остановился раньше лимита шагов
    assert "недоступен" in out["final"]["notes"]
    assert out["final"]["outliers_count"] is None  # не выдумал число


def test_loop_respects_max_steps(ctx, tmp_path):
    """Жёсткий лимит шагов ограничивает цикл даже при «вечном» планировании."""
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg)
    # агент бесконечно вызывает корректный инструмент — упрёмся в max_steps
    action = {"thought": "x", "tool": "get_curve_info", "args": {"curve_id": "b001"}}
    llm = FakeLLM([action])
    sess = AgentSession({"id": "t_steps", "curve_id": "b001", "query": "?", },
                        ds, cfg, llm=llm, executor=ex, max_steps=5,
                        log_dir=str(tmp_path))
    out = sess.run()
    assert out["stats"]["steps"] == 5
    assert out["stats"]["degraded"] is True
    assert "лимит шагов" in out["final"]["notes"]


def test_loop_enforces_min_tool_calls(ctx, tmp_path):
    """Финал без единого вызова инструмента отклоняется один раз."""
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg)
    llm = FakeLLM([
        {"final": True, "periodic": True, "period_days": 3.2, "period_unit": "days",
         "outliers_count": 0, "outliers_indices": [], "flares_count": None,
         "dips_count": None, "notes": "периодична"},
        {"thought": "x", "tool": "get_curve_info", "args": {"curve_id": "b001"}},
        {"final": True, "periodic": True, "period_days": 3.2, "period_unit": "days",
         "outliers_count": 0, "outliers_indices": [], "flares_count": None,
         "dips_count": None, "notes": "периодична"},
    ])
    sess = AgentSession({"id": "t_min", "curve_id": "b001", "query": "?", },
                        ds, cfg, llm=llm, executor=ex, max_steps=10,
                        log_dir=str(tmp_path))
    out = sess.run()
    assert out["stats"]["tool_calls"] == 1  # наг исправил поведение
    assert out["final"]["periodic"] is True
    assert out["stats"]["degraded"] is False