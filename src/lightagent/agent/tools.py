"""Реестр инструментов агента: схемы, валидация аргументов, обработчики.

Принципы:
- инструменты только читают назначенный датасет и пишут отчёты/графики;
  исполнения произвольного кода агенту не даётся (граница автономности);
- каждый вызов проходит локальную валидацию по схеме (типы, диапазоны,
  enum) — контроль аргументов, который LLM обойти не может;
- семантические ошибки (период вне наблюдаемого диапазона, неизвестная
  кривая, невозможный time_unit) возвращаются как понятные ошибки,
  на которые планировщик может отреагировать и исправить вызов.

Ground truth (data/ground_truth.json) инструментам НЕ передаётся: агент
работает только по наблюдаемым данным.
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable

import numpy as np

from ..analysis import anomalies as an
from ..analysis import period as per
from ..config import Config
from ..data.generator import Curve
from ..data.loader import Dataset

T = Callable[..., dict]


class ToolValidationError(Exception):
    """Невалидные аргументы вызова (до выполнения)."""


class ToolSemanticError(Exception):
    """Семантическая ошибка инструмента (корректная схема, плохие данные)."""


def _check(spec: dict, args: dict):
    """Локальная валидация args по spec. Бросает ToolValidationError."""
    if not isinstance(args, dict):
        raise ToolValidationError(f"аргументы должны быть JSON-объектом, получено {type(args).__name__}")
    props = spec.get("properties", {})
    for name in list(args):
        if name not in props:
            raise ToolValidationError(
                f"неизвестный аргумент {name!r}; допустимые: {sorted(props)}")
    for name, p in props.items():
        if name in args and args[name] is None:
            raise ToolValidationError(f"аргумент {name!r} не может быть null")
        if p.get("required") and name not in args:
            raise ToolValidationError(f"обязательный аргумент {name!r} отсутствует")
        if name not in args:
            continue
        val = args[name]
        typ = p.get("type", "string")
        ok_type = {
            "string": isinstance(val, str),
            "number": isinstance(val, (int, float)) and not isinstance(val, bool),
            "integer": isinstance(val, int) and not isinstance(val, bool),
            "boolean": isinstance(val, bool),
            "array": isinstance(val, list),
        }[typ]
        if not ok_type:
            raise ToolValidationError(
                f"аргумент {name!r}: ожидался {typ}, получено {type(val).__name__} = {val!r}")
        if "enum" in p and val not in p["enum"]:
            raise ToolValidationError(
                f"аргумент {name!r}={val!r}: допустимо только {p['enum']}")
        if isinstance(val, (int, float)):
            if "minimum" in p and val < p["minimum"]:
                raise ToolValidationError(
                    f"аргумент {name!r}={val}: минимум {p['minimum']}")
            if "maximum" in p and val > p["maximum"]:
                raise ToolValidationError(
                    f"аргумент {name!r}={val}: максимум {p['maximum']}")


def _req_curve(ds: Dataset, curve_id: str) -> Curve:
    try:
        return ds.get(curve_id)
    except KeyError as exc:
        raise ToolSemanticError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Обработчики
# ---------------------------------------------------------------------------

def h_list_curves(ds: Dataset, cfg: Config, args: dict) -> dict:
    rows = []
    for cid in ds.ids():
        c = ds.get(cid)
        st = an.series_stats(c.t, c.flux, c.flux_err)
        rows.append({"curve_id": cid, "n_points": st["n_points"],
                     "span_days": round(st["span_days"], 2),
                     "median_dt_days": round(st["median_dt_days"], 3)})
    return {"count": len(rows), "curves": rows}


def h_get_curve_info(ds: Dataset, cfg: Config, args: dict) -> dict:
    c = _req_curve(ds, args["curve_id"])
    st = an.series_stats(c.t, c.flux, c.flux_err)
    st["curve_id"] = args["curve_id"]
    st["note"] = ("время в сутках (MJD-подобные); поток в относительных "
                  "единицах, среднее ~1; пропуски = сезонные окна + случайные")
    return st


def h_periodogram(ds: Dataset, cfg: Config, args: dict) -> dict:
    a = cfg.section("analysis")
    c = _req_curve(ds, args["curve_id"])
    t, y, dy = c.t, c.flux, c.flux_err
    unit = args.get("time_unit", "days")
    freq_min = args.get("freq_min")
    freq_max = args.get("freq_max")
    n_peaks = int(args.get("n_peaks", a["n_peaks"]))
    spp = int(args.get("samples_per_peak", a["samples_per_peak"]))
    if freq_min is not None and freq_max is not None and freq_min >= freq_max:
        raise ToolSemanticError(
            f"freq_min={freq_min} >= freq_max={freq_max}; диапазон частот пуст")
    sigma = an._robust_scale(y)[1]
    if sigma < 1e-9 * (abs(float(np.mean(y))) + 1.0):
        raise ToolSemanticError(
            "дисперсия ряда практически нулевая (постоянная кривая?): "
            "поиск периода не имеет смысла, периодограмма не строится")

    # Стандартная астрономическая практика: линейный тренд снимается ДО
    # периодограммы — иначе монотонный дрейф создаёт ложный низкочастотный
    # пик с маленьким FAP (классическая «ложная периодичность»).
    coef = np.polyfit(t, y, 1)
    y_dt = y - np.polyval(coef, t)
    peaks, meta = per.lomb_scargle_peaks(
        t, y_dt, dy, samples_per_peak=spp, n_peaks=n_peaks,
        freq_min=freq_min, freq_max=freq_max)
    meta["detrended"] = True
    meta["linear_slope_per_day"] = float(coef[0])
    meta["linear_intercept"] = float(coef[1])
    thr = float(a["fap_threshold"])
    span = float(np.ptp(t))
    df = 1.0 / max(span, 1e-9)
    out_peaks = []
    top_sig = None
    for rank, pk in enumerate(peaks, start=1):
        p_days = pk["period_days"]
        res = {
            "rank": rank,
            "period_days": round(p_days, 6) if p_days else None,
            f"period_{unit}": round(per.convert_period(p_days, unit), 6),
            "freq_1_per_day": round(pk["freq_1_per_day"], 8),
            "power": round(pk["power"], 4),
            "fap": f"{pk['fap']:.2e}",
            "significant": bool(pk["fap"] <= thr),
        }
        # краевой пик (у границы диапазона частот) — ненадёжен
        f_pk = pk["freq_1_per_day"]
        if f_pk < meta["freq_min"] + 2 * df or f_pk > meta["freq_max"] - 2 * df:
            res["edge_peak"] = True
            res["edge_note"] = ("пик на краю диапазона частот: ненадёжен "
                                "как кандидат в период")
        # гармоники: мощность на 2f и на субгармонике f/2
        if p_days:
            ls, _ = per._get_ls(t, y_dt, dy)
            p2 = float(ls.power([2.0 / p_days])[0])
            ratio2 = p2 / max(pk["power"], 1e-12)
            res["harmonic_period_days"] = round(p_days / 2.0, 6)
            res["harmonic_power_ratio"] = round(ratio2, 3)
            res["harmonic_similar"] = bool(ratio2 >= float(a["harmonic_power_ratio"]))
            if res["harmonic_similar"]:
                res["harmonic_note"] = (
                    f"пик на {p_days:.4f} сут сравним с гармоникой на "
                    f"{p_days/2:.4f} сут (отношение мощностей {ratio2:.2f}): "
                    f"фундаментальный период может быть 2×{p_days:.4f} сут")
            # субгармоника (f/2): признак «два провала/горба за цикл»
            p_sub = float(ls.power([0.5 / p_days])[0])
            ratio_sub = p_sub / max(pk["power"], 1e-12)
            res["subharmonic_period_days"] = round(2.0 * p_days, 6)
            res["subharmonic_power_ratio"] = round(ratio_sub, 3)
            res["subharmonic_similar"] = bool(ratio_sub >= 0.35)
            if res["subharmonic_similar"]:
                res["subharmonic_note"] = (
                    f"у пика {p_days:.4f} сут сильна субгармоника на "
                    f"{2*p_days:.4f} сут (отношение {ratio_sub:.2f}): типично "
                    f"для затменной двойной — два провала за цикл, "
                    f"фундаментальный период может быть 2×{p_days:.4f} сут")
            # суточный алиасинг
            w_1d = per.window_function_power(t, 1.0)
            if abs(1.0 / p_days - 1.0) <= float(a["one_day_alias_tol_d"]) and w_1d > 0.5:
                res["one_day_alias_note"] = (
                    f"кандидат {p_days:.4f} сут близок к 1 сут, окно сильно "
                    f"на 1 сут (W={w_1d:.2f}): возможен суточный алиасинг")
            res["window_1d_power"] = round(w_1d, 3)
        out_peaks.append(res)
        if res["significant"] and top_sig is None:
            top_sig = res
    unit_factor = per.TIME_UNITS[unit]
    return {
        "curve_id": args["curve_id"],
        "time_unit": unit,
        "fap_threshold": thr,
        "freq_range_1_per_day": [round(meta["freq_min"], 6), round(meta["freq_max"], 6)],
        "n_freq": meta["n_freq"],
        "period_range_days": [round(1.0 / meta["freq_max"], 6), round(1.0 / meta["freq_min"], 6)],
        "span_days": round(float(np.ptp(t)), 2),
        "detrended": True,
        "linear_slope_per_day": round(meta["linear_slope_per_day"], 8),
        "note_unit": f"периоды приведены в {unit}: period_{unit} = period_days × {unit_factor}",
        "note_detrend": "линейный тренд снят перед периодограммой; наклон "
                        "линейной аппроксимации — linear_slope_per_day",
        "peaks": out_peaks,
        "recommended": top_sig or None,
    }


def h_check_alias(ds: Dataset, cfg: Config, args: dict) -> dict:
    a = cfg.section("analysis")
    c = _req_curve(ds, args["curve_id"])
    t = c.t
    unit = args.get("time_unit", "days")
    p_days = per.convert_period(float(args["period"]), unit) / per.TIME_UNITS[unit] \
        if args.get("period_unit") else float(args["period"])
    # период может быть передан в любой единице; по умолчанию — в сутках
    p_unit = args.get("period_unit", "days")
    p_days = per.convert_period(float(args["period"]), p_unit)
    warnings, flags = per.check_period_aliases(
        t, p_days, signal_power=1.0,
        harmonic_power_ratio=float(a["harmonic_power_ratio"]),
        one_day_alias_tol_d=float(a["one_day_alias_tol_d"]))
    f = 1.0 / p_days
    # мощности окна на интересных частотах
    w_cand = per.window_function_power(t, f)
    w_1d = per.window_function_power(t, 1.0)
    return {
        "curve_id": args["curve_id"],
        "period_days": round(p_days, 6),
        "window_power_at_candidate": round(w_cand, 3),
        "window_power_at_1day": round(w_1d, 3),
        "warnings": warnings,
    }


def h_fit_linear_trend(ds: Dataset, cfg: Config, args: dict) -> dict:
    c = _req_curve(ds, args["curve_id"])
    t, y, dy = c.t, c.flux, c.flux_err
    if len(t) < 3:
        raise ToolSemanticError("слишком мало точек для оценки тренда")
    w = 1.0 / np.maximum(dy, 1e-12) ** 2
    coef, cov = np.polyfit(t, y, 1, w=w, cov=True)
    slope_std = float(np.sqrt(cov[0, 0])) if cov is not None else float("nan")
    if slope_std > 0:
        significance = abs(float(coef[0])) / slope_std
    else:
        significance = float("nan")
    resid = y - np.polyval(coef, t)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "curve_id": args["curve_id"],
        "slope_per_day": float(coef[0]),
        "slope_std": slope_std,
        "significance": significance,
        "rms_after": rms,
        "trend_present": bool(significance > 3.0),
    }


def h_detect_outliers(ds: Dataset, cfg: Config, args: dict) -> dict:
    a = cfg.section("analysis")
    c = _req_curve(ds, args["curve_id"])
    threshold = args.get("threshold", a["outlier"]["threshold"])
    max_frac = args.get("max_fraction", a["outlier"]["max_fraction"])
    res = an.detect_outliers(c.flux, c.flux_err, threshold=threshold,
                             max_fraction=max_frac)
    res["curve_id"] = args["curve_id"]
    res["indices"] = res["indices"][:500]  # защита объёма вывода
    return res


def h_detect_flares(ds: Dataset, cfg: Config, args: dict) -> dict:
    a = cfg.section("analysis")
    c = _req_curve(ds, args["curve_id"])
    res = an.detect_flares(c.t, c.flux, c.flux_err,
                           significance=args.get("significance", a["flare"]["significance"]),
                           min_points=args.get("min_points", a["flare"]["min_points"]))
    res["curve_id"] = args["curve_id"]
    res["flares"] = res["flares"][:10]
    return res


def h_detect_dips(ds: Dataset, cfg: Config, args: dict) -> dict:
    a = cfg.section("analysis")
    c = _req_curve(ds, args["curve_id"])
    res = an.detect_dips(c.t, c.flux, c.flux_err,
                         significance=args.get("significance", a["dip"]["significance"]),
                         min_points=args.get("min_points", a["dip"]["min_points"]))
    res["curve_id"] = args["curve_id"]
    res["dips"] = res["dips"][:10]
    return res


def h_plot_curve(ds: Dataset, cfg: Config, args: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = _req_curve(ds, args["curve_id"])
    out_dir = pathlib.Path(args.get("out_dir") or (cfg.root / "report" / "plots"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args['curve_id']}.png"
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.errorbar(c.t, c.flux, yerr=c.flux_err, fmt=".", ms=3, lw=0.8,
                alpha=0.8, color="#1f6feb")
    ax.set_xlabel("время, сутки")
    ax.set_ylabel("поток (отн.)")
    ax.set_title(f"Кривая блеска {args['curve_id']}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {"curve_id": args["curve_id"], "plot_path": str(path),
            "note": "график сохранён для отчёта; аналитические выводы график не заменяет"}


# ---------------------------------------------------------------------------
# Реестр
# ---------------------------------------------------------------------------

def build_specs() -> dict[str, dict]:
    return {
        "list_curves": {
            "description": "Список доступных кривых с числом точек, длительностью "
                           "и медианным интервалом наблюдений. Без аргументов.",
            "properties": {},
        },
        "get_curve_info": {
            "description": "Статистика кривой: число точек, покрытие, пропуски "
                           "(макс. разрыв, число разрывов >5 сут), шум, вариабельность. "
                           "Время в сутках. Вызывай перед анализом.",
            "properties": {
                "curve_id": {"type": "string", "required": True,
                             "description": "идентификатор кривой"},
            },
        },
        "periodogram": {
            "description": "Периодограмма Ломба-Скаргла. Возвращает топ пиков с "
                           "периодом (в time_unit), мощностью и FAP. Пик значим при "
                           "fap <= 1e-3 (см. fap_threshold). Помечает гармоники и "
                           "суточный алиасинг. Единицы времени: days,hours,minutes,seconds.",
            "properties": {
                "curve_id": {"type": "string", "required": True},
                "freq_min": {"type": "number", "description": "мин. частота, 1/сутки",
                             "minimum": 1e-6},
                "freq_max": {"type": "number", "description": "макс. частота, 1/сутки",
                             "minimum": 1e-6},
                "samples_per_peak": {"type": "integer", "minimum": 3, "maximum": 50,
                                     "description": "плотность сетки (по умолчанию 10)"},
                "n_peaks": {"type": "integer", "minimum": 1, "maximum": 8},
                "time_unit": {"type": "string", "enum": ["days", "hours", "minutes", "seconds"],
                              "description": "единица возвращаемых периодов"},
            },
        },
        "check_alias": {
            "description": "Проверка конкретного кандидата в период на гармоники и "
                           "суточный алиасинг (спектральное окно). period задаётся "
                           "в сутках (period_unit='days') или другой единице.",
            "properties": {
                "curve_id": {"type": "string", "required": True},
                "period": {"type": "number", "required": True, "minimum": 1e-4,
                           "description": "значение периода"},
                "period_unit": {"type": "string",
                                "enum": ["days", "hours", "minutes", "seconds"],
                                "description": "единица period (по умолчанию days)"},
                "time_unit": {"type": "string",
                              "enum": ["days", "hours", "minutes", "seconds"]},
            },
        },
        "fit_linear_trend": {
            "description": "Линейная аппроксимация потока по времени (взвешенная "
                           "по ошибкам): наклон, ошибка наклона, значимость. "
                           "trend_present=true при |slope|/std > 3.",
            "properties": {
                "curve_id": {"type": "string", "required": True},
            },
        },
        "detect_outliers": {
            "description": "Поиск выбросов по MAD-методу: |y-med|/(1.4826·MAD) > threshold "
                           "и отклонение > 3·flux_err. Число выбросов ограничено "
                           "max_fraction·N. Возвращает индексы и долю.",
            "properties": {
                "curve_id": {"type": "string", "required": True},
                "threshold": {"type": "number", "minimum": 2.0, "maximum": 20.0,
                              "description": "порог MAD-отклонения (по умолчанию 5)"},
                "max_fraction": {"type": "number", "minimum": 0.01, "maximum": 0.5},
            },
        },
        "detect_flares": {
            "description": "Поиск вспышек: подряд >= min_points точек выше базового "
                           "уровня на more чем significance шумовых единиц. Возвращает "
                           "интервалы, пиковое время и амплитуду.",
            "properties": {
                "curve_id": {"type": "string", "required": True},
                "significance": {"type": "number", "minimum": 2.0, "maximum": 15.0},
                "min_points": {"type": "integer", "minimum": 2, "maximum": 20},
            },
        },
        "detect_dips": {
            "description": "Поиск провалов (затмения/транзиты): подряд >= min_points "
                           "точек ниже базового уровня на more чем significance "
                           "шумовых единиц. Возвращает интервалы и глубину.",
            "properties": {
                "curve_id": {"type": "string", "required": True},
                "significance": {"type": "number", "minimum": 2.0, "maximum": 15.0},
                "min_points": {"type": "integer", "minimum": 2, "maximum": 20},
            },
        },
        "plot_curve": {
            "description": "Сохранить график кривой в report/plots/ и вернуть путь. "
                           "Для отчёта; выводы делай по численным инструментам.",
            "properties": {
                "curve_id": {"type": "string", "required": True},
            },
        },
    }


class ToolRegistry:
    def __init__(self, ds: Dataset, cfg: Config):
        self.ds = ds
        self.cfg = cfg
        self.specs = build_specs()
        self.handlers = {
            "list_curves": h_list_curves,
            "get_curve_info": h_get_curve_info,
            "periodogram": h_periodogram,
            "check_alias": h_check_alias,
            "fit_linear_trend": h_fit_linear_trend,
            "detect_outliers": h_detect_outliers,
            "detect_flares": h_detect_flares,
            "detect_dips": h_detect_dips,
            "plot_curve": h_plot_curve,
        }

    def available(self):
        return sorted(self.handlers)

    def spec_text(self) -> str:
        return json.dumps(self.specs, ensure_ascii=False, indent=1)

    def validate_call(self, name: str, args: dict):
        """Локальный контроль аргументов; бросает ToolValidationError."""
        if name not in self.handlers:
            raise ToolValidationError(
                f"неизвестный инструмент {name!r}; доступны: {self.available()}")
        _check(self.specs[name], args)

    def run(self, name: str, args: dict) -> dict:
        """Выполнить инструмент (вызывается только после validate_call)."""
        return self.handlers[name](self.ds, self.cfg, args)