"""Тесты численных методов и генератора."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from lightagent.config import load_config
from lightagent.data.loader import load_dataset
from lightagent.analysis import anomalies as an
from lightagent.analysis import period as per


@pytest.fixture(scope="module")
def ds():
    cfg = load_config()
    d = cfg.section("dataset")
    return load_dataset(cfg.root / d["out_csv"], cfg.root / d["out_truth"])


def test_generator_reproducible(tmp_path):
    """Одна и та же кривая детерминирована."""
    from lightagent.data.generator import generate_curve
    c1, _ = generate_curve("b000", "periodic")
    c2, _ = generate_curve("b000", "periodic")
    assert np.allclose(c1.t, c2.t) and np.allclose(c1.flux, c2.flux)


def test_sampling_is_nonuniform_with_gaps(ds):
    """Неравномерная дискретизация и пропуски: dt меняется, есть разрывы."""
    for cid in ["b000", "b021", "b005"]:
        c = ds.get(cid)
        dt = np.diff(c.t)
        assert dt.min() > 0
        assert dt.std() / (dt.mean() + 1e-9) > 0.2, "нет джиттера каденса"
        assert dt.max() > 10 * dt.mean(), "нет крупных пропусков"


def test_lomb_scargle_recovers_period(ds):
    cfg = load_config()
    a = cfg.section("analysis")
    for cid in ["b000", "b009", "b032", "b035"]:
        c = ds.get(cid)
        true_p = ds.truths[cid]["P_true"]
        coef = np.polyfit(c.t, c.flux, 1)
        y = c.flux - np.polyval(coef, c.t)
        peaks, _ = per.lomb_scargle_peaks(c.t, y, c.flux_err,
                                          samples_per_peak=a["samples_per_peak"],
                                          n_peaks=a["n_peaks"])
        sig = [p for p in peaks if p["fap"] <= a["fap_threshold"]]
        assert sig, f"{cid}: значимого пика нет"
        err = abs(sig[0]["period_days"] - true_p) / true_p
        assert err < 0.05, f"{cid}: ошибка периода {err:.4f}"


def test_no_false_periodicity_after_detrend(ds):
    """Спокойные и трендовые кривые не дают значимых пиков."""
    cfg = load_config()
    a = cfg.section("analysis")
    for cid in ["b002", "b005", "b010", "b018", "b033", "b039"]:
        c = ds.get(cid)
        coef = np.polyfit(c.t, c.flux, 1)
        y = c.flux - np.polyval(coef, c.t)
        peaks, _ = per.lomb_scargle_peaks(c.t, y, c.flux_err,
                                          samples_per_peak=a["samples_per_peak"],
                                          n_peaks=a["n_peaks"])
        assert not [p for p in peaks if p["fap"] <= a["fap_threshold"]], cid


def test_outliers_detected(ds):
    cfg = load_config()
    a = cfg.section("analysis")
    for cid in ["b001", "b012", "b014", "b015"]:
        c = ds.get(cid)
        tr = ds.truths[cid]
        res = an.detect_outliers(c.flux, c.flux_err,
                                 threshold=a["outlier"]["threshold"],
                                 max_fraction=a["outlier"]["max_fraction"])
        true_set = set(tr["outliers_indices"])
        pred_set = set(res["indices"])
        assert true_set == pred_set, f"{cid}: {true_set - pred_set} пропущено, лишние {(pred_set - true_set) or set()}"


def test_flares_and_dips_detected(ds):
    cfg = load_config()
    a = cfg.section("analysis")
    c = ds.get("b021")
    f = an.detect_flares(c.t, c.flux, c.flux_err,
                         significance=a["flare"]["significance"],
                         min_points=a["flare"]["min_points"])
    assert f["count"] > 0 and len(ds.truths["b021"]["flares"]) > 0
    c = ds.get("b038")
    d = an.detect_dips(c.t, c.flux, c.flux_err,
                       significance=a["dip"]["significance"],
                       min_points=a["dip"]["min_points"])
    assert d["count"] > 0 and len(ds.truths["b038"]["dips"]) > 0


def test_convert_period_units():
    assert per.convert_period(2.0, "hours") == 48.0
    assert per.convert_period(1.0, "minutes") == 1440.0
    assert per.convert_period(1.0, "seconds") == 86400.0
    with pytest.raises(ValueError):
        per.convert_period(1.0, "parsec")


def test_ground_truth_consistent(ds):
    """Тип из ground truth согласован с полями."""
    for cid, tr in ds.truths.items():
        typ = tr["type"]
        assert tr["curve_id"] == cid
        if typ == "periodic" or typ == "eclipsing":
            assert tr.get("P_true") and tr["P_true"] > 0
        else:
            assert not tr.get("P_true")
        for key in ("outliers_indices", "flares", "dips"):
            assert isinstance(tr.get(key, []), list)