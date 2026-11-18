"""Тесты метрик сравнения."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lightagent.metrics import _set_f1, evaluate_task, aggregate


def test_set_f1():
    assert _set_f1([10, 20], [], tol=2) == 0.0
    assert _set_f1([], [], tol=2) == 1.0
    assert _set_f1([10, 11], [10, 11], tol=2) == 1.0
    # допуск соседства: 100 найден как 101
    assert _set_f1([101], [100], tol=2) == 1.0
    assert 0.0 < _set_f1([10, 50], [10, 51], tol=1) <= 1.0


def test_period_hit_none_when_not_periodic():
    task = {"id": "x", "curve_id": "b010", "expected": {"periodic": False}}
    res = {"final": {"periodic": False, "period_days": None, "outliers_indices": []},
           "stats": {}}
    m = evaluate_task(task, res, task["expected"])
    assert m["period_hit"] is None
    assert m["periodic_correct"] is True


def test_period_hit_tolerance():
    task = {"id": "x", "curve_id": "b000",
            "expected": {"periodic": True, "period_days": 10.0}}
    res = {"final": {"periodic": True, "period_days": 10.3, "outliers_indices": []},
           "stats": {}}
    m = evaluate_task(task, res, task["expected"])
    assert m["period_hit"] is True
    res2 = {"final": {"periodic": True, "period_days": 12.0, "outliers_indices": []},
            "stats": {}}
    assert evaluate_task(task, res2, task["expected"])["period_hit"] is False


def test_outliers_f1_only_when_expected():
    task = {"id": "x", "curve_id": "b021",
            "expected": {"periodic": False, "outliers_indices": []}}
    res = {"final": {"periodic": False, "outliers_indices": [5, 9]}, "stats": {}}
    m = evaluate_task(task, res, task["expected"])
    assert m["outliers_f1"] is None  # выбросы не запрашивались


def test_caveat_hit():
    task = {"id": "x", "curve_id": "b005",
            "expected": {"periodic": False, "caveat_required": True}}
    res = {"final": {"periodic": False,
                     "notes": "тренд снят, пропуски учтены"}, "stats": {}}
    m = evaluate_task(task, res, task["expected"])
    assert m["caveat_hit"] is True


def test_aggregate_counts():
    rows = [
        {"periodic_correct": True, "periodic_decided": True, "period_hit": True,
         "outliers_f1": 1.0, "success": True, "tool_calls": 5,
         "est_cost_usd": 0.01},
        {"periodic_correct": False, "periodic_decided": True, "period_hit": None,
         "outliers_f1": None, "success": False, "tool_calls": 7,
         "est_cost_usd": 0.02},
    ]
    agg = aggregate(rows)
    assert agg["periodic_acc"] == 0.5
    assert agg["period_hit_rate"] == 1.0
    assert agg["outliers_f1_mean"] == 1.0
    assert agg["success_rate"] == 0.5