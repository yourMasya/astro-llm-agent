"""Анализ периодичности: Lomb-Scargle периодограмма, FAP, гармоники, алиасы.

Основной инструмент — astropy.timeseries.LombScargle (устойчива к пропускам и
неравномерной дискретизации). Если astropy недоступна, используется лёгкая
numpy-реализация того же периодограммного статистика; FAP — аналитическая
оценка Балёва (Baluev 2008) или бутстрэп.
"""

from __future__ import annotations

import numpy as np

TIME_UNITS = {"days": 1.0, "hours": 24.0, "minutes": 1440.0, "seconds": 86400.0}


def _lomb_scargle_impl():
    """Возвращает класс LombScargle из astropy или лёгкий fallback."""
    try:
        from astropy.timeseries import LombScargle
        return LombScargle, "astropy"
    except Exception:  # pragma: no cover
        return None, "fallback"


class _FallbackLS:
    """Минимальная numpy-реализация стандартной (нормированной) LS."""

    def __init__(self, t, y, dy):
        self.t = np.asarray(t, float)
        self.y = np.asarray(y, float)
        self.dy = np.asarray(dy, float)
        w = 1.0 / np.maximum(self.dy, 1e-12) ** 2
        self.w = w / w.sum()

    def _center(self, f):
        omega = 2 * np.pi * f
        x = omega * self.t
        sw = np.sin(2 * x) * self.w
        cw = np.cos(2 * x) * self.w
        tau = 0.5 * np.arctan2(sw.sum(), cw.sum()) / omega
        return x - omega * tau

    def power(self, frequency):
        f = np.atleast_1d(frequency)
        out = np.empty(len(f))
        yw = self.y - (self.w * self.y).sum()
        for i, fi in enumerate(f):
            x = self._center(fi)
            s, c = np.sin(x), np.cos(x)
            ys, yc = (yw * s).sum(), (yw * c).sum()
            ss, cc = (self.w * s * s).sum(), (self.w * c * c).sum()
            sc = (self.w * s * c).sum()
            denom = ss * cc - sc * sc
            out[i] = (yc * yc * ss + ys * ys * cc - 2 * yc * ys * sc) / max(denom, 1e-300)
        return out

    @staticmethod
    def false_alarm_probability(power, method="baluev", N_eff=None):
        # Аппроксимация Балёва: FAP ~ 1 - exp(-tau * exp(-power)).
        # Для fallback значение tau подбирается по числу независимых частот.
        z = np.asarray(power, float)
        n_eff = N_eff or 2000
        tau = np.sqrt(4 * n_eff)  # грубая оценка
        return 1.0 - np.exp(-tau * np.exp(-z))


def _get_ls(t, y, dy):
    cls, kind = _lomb_scargle_impl()
    if cls is None:
        return _FallbackLS(t, y, dy), "fallback"
    return cls(t, y, dy), "astropy"


def frequency_grid(t, samples_per_peak=10, freq_min=None, freq_max=None):
    """Частотная сетка: от ~2 полных циклов за span до псевдо-Найквиста."""
    span = np.ptp(t)
    med_dt = float(np.median(np.diff(np.sort(t))))
    if freq_min is None:
        freq_min = 2.0 / max(span, 1e-9)
    if freq_max is None:
        freq_max = 1.0 / (2.0 * max(med_dt, 1e-9))
    # обычная практика: не более samples_per_peak точек на пик
    df = 1.0 / max(span, 1e-9)
    n = int(max(10, (freq_max - freq_min) / df * samples_per_peak))
    freqs = np.linspace(freq_min, freq_max, n)
    return freqs, freq_min, freq_max


def _peaks(freqs, power, n_peaks=5, min_sep_frac=0.02):
    """Локальные максимумы, отсортированные по мощности, с разведением."""
    if len(power) < 3:
        return []
    order = min(3, len(power) // 4 or 1)
    is_max = np.ones(len(power), bool)
    for i in range(len(power)):
        lo = max(0, i - order)
        hi = min(len(power), i + order + 1)
        if power[i] != power[lo:hi].max():
            is_max[i] = False
    idx = np.where(is_max)[0]
    idx = idx[np.argsort(power[idx])[::-1]]
    span = max(freqs[-1] - freqs[0], 1e-9)
    sep = min_sep_frac * (freqs[-1] - freqs[0])
    picked = []
    for i in idx:
        if all(abs(freqs[i] - freqs[j]) > sep for j in picked):
            picked.append(int(i))
        if len(picked) >= n_peaks:
            break
    return sorted(picked, key=lambda j: -power[j])


def lomb_scargle_peaks(t, y, dy, *, samples_per_peak=10, n_peaks=5,
                       freq_min=None, freq_max=None,
                       fap_method="baluev", rng=None):
    """Топ пиков периодограммы: [{freq, period_days, power, fap}]."""
    ls, kind = _get_ls(t, y, dy)
    freqs, fmin, fmax = frequency_grid(t, samples_per_peak, freq_min, freq_max)
    power = ls.power(freqs)
    peaks_idx = _peaks(freqs, power, n_peaks)
    res = []
    for j in peaks_idx:
        fap = ls.false_alarm_probability(power[j], method=fap_method)
        res.append({
            "freq_1_per_day": float(freqs[j]),
            "period_days": float(1.0 / freqs[j]) if freqs[j] > 0 else None,
            "power": float(power[j]),
            "fap": float(fap),
        })
    meta = {"impl": kind, "freq_min": float(fmin), "freq_max": float(fmax),
            "n_freq": int(len(freqs))}
    return res, meta


def window_function_power(t, f_1_per_day):
    """Мощность спектрального окна на частоте f (нормированная 0..1)."""
    x = 2 * np.pi * np.asarray(f_1_per_day) * t
    w = (np.exp(1j * x)).sum()
    return float(np.abs(w) ** 2 / len(t))


def check_period_aliases(t, period_days, signal_power, *,
                         harmonic_power_ratio=0.8,
                         one_day_alias_tol_d=0.01,
                         harmonic_power=None):
    """Проверка кандидата на гармоники и суточный алиасинг.

    Возвращает список предупреждений (строками) и структурированные признаки.
    Логика:
    - если пик на 2f (P/2) почти такой же силы, период может быть гармоникой;
    - если спектральное окно сильно на 1 сутки, пик около 1 суток подозрителен.
    """
    warnings = []
    flags = {"harmonic": False, "one_day_alias": False}
    f = 1.0 / period_days if period_days > 0 else None
    if f is None:
        return warnings, flags

    if harmonic_power is not None and signal_power > 0:
        if harmonic_power / max(signal_power, 1e-12) >= harmonic_power_ratio:
            warnings.append(
                f"пик на {period_days:.4f} сут сравним с пиком на "
                f"{period_days/2:.4f} сут (2f): это может быть вторая гармоника, "
                f"фундаментальный период может быть вдвое больше")
            flags["harmonic"] = True

    w_1d = window_function_power(t, 1.0)
    if abs(f - 1.0) <= one_day_alias_tol_d and w_1d > 0.5:
        warnings.append(
            f"период {period_days:.4f} сут подозрительно близок к 1 суткам, "
            f"а спектральное окно сильно на 1 сут (W={w_1d:.2f}): возможен "
            f"суточный алиасинг — проверьте независимо")
        flags["one_day_alias"] = True
    flags["window_1d_power"] = w_1d
    return warnings, flags


def convert_period(period_days: float, unit: str) -> float:
    """Перевод периода из суток в запрошенную единицу времени."""
    if unit not in TIME_UNITS:
        raise ValueError(f"неизвестная единица {unit!r}; допустимо {sorted(TIME_UNITS)}")
    return period_days * TIME_UNITS[unit]