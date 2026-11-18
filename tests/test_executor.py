"""Тесты исполнителя: валидация аргументов, ретраи, ошибки."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from lightagent.config import load_config
from lightagent.data.loader import load_dataset
from lightagent.agent.executor import ToolExecutor


@pytest.fixture(scope="module")
def ctx():
    cfg = load_config()
    d = cfg.section("dataset")
    ds = load_dataset(cfg.root / d["out_csv"], cfg.root / d["out_truth"])
    return cfg, ds


def test_unknown_tool(ctx):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg)
    r = ex.call("hack", {})
    assert r.status == "validation_error"
    assert "неизвестный инструмент" in r.error


def test_wrong_type(ctx):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg)
    r = ex.call("periodogram", {"curve_id": 42})
    assert r.status == "validation_error"
    assert "ожидался string" in r.error


def test_out_of_range(ctx):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg)
    r = ex.call("detect_outliers", {"curve_id": "b001", "threshold": 0.5})
    assert r.status == "validation_error"
    r = ex.call("detect_outliers", {"curve_id": "b001", "threshold": 50})
    assert r.status == "validation_error"


def test_unknown_argument_name(ctx):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg)
    r = ex.call("detect_outliers", {"curve_id": "b001", "threshhold": 5.0})
    assert r.status == "validation_error"
    assert "неизвестный аргумент" in r.error


def test_unknown_curve_id_semantic(ctx):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg)
    r = ex.call("get_curve_info", {"curve_id": "b999"})
    assert r.status == "semantic_error"
    assert "не найдена" in r.error


def test_transient_retry(ctx):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg, retries_transient=2, retry_backoff_s=(0.0, 0.0))
    ex.inject_failure("periodogram", lambda: TimeoutError("boom"), attempts=2)
    r = ex.call("periodogram", {"curve_id": "b000", "time_unit": "days"})
    assert r.ok and r.retries == 2


def test_persistent_failure_no_retry_exhaustion(ctx):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg, retries_transient=2, retry_backoff_s=(0.0, 0.0))
    ex.inject_failure("detect_outliers", lambda: RuntimeError("down"), attempts=99)
    r = ex.call("detect_outliers", {"curve_id": "b001"})
    assert r.status == "error"
    assert "RuntimeError" in r.error
    assert r.retries == 0  # не транзиентная: повторять не нужно


def test_semantic_not_retried(ctx):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg, retries_transient=2, retry_backoff_s=(0.0, 0.0))
    r = ex.call("periodogram", {"curve_id": "b000", "freq_min": 5, "freq_max": 1})
    assert r.status == "semantic_error"
    r = ex.call("periodogram", {"curve_id": "b000", "time_unit": "parsec"})
    assert r.status == "validation_error"


def test_constant_curve_rejected(ctx):
    cfg, ds = ctx
    from lightagent.data.generator import Curve
    import numpy as np
    from lightagent.data.loader import Dataset
    n = 50
    ds2 = Dataset({"flat": Curve(id="flat", t=np.arange(n, dtype=float),
                                 flux=np.ones(n), flux_err=np.full(n, 0.01))})
    ex = ToolExecutor(ds2, cfg)
    r = ex.call("periodogram", {"curve_id": "flat", "time_unit": "days"})
    assert r.status == "semantic_error"
    assert "дисперсия" in r.error


def test_result_truncation(ctx, tmp_path):
    cfg, ds = ctx
    ex = ToolExecutor(ds, cfg, obs_char_limit=300)
    r = ex.call("list_curves", {})
    assert len(r.text) <= 300
    assert r.ok