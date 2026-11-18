"""Обнаружение аномалий: выбросы, вспышки, провалы (затмения/транзиты).

Все методы робастные (медиана/MAD), с порогами в относительных шумовых
единицах и защитой от абсурдных результатов (кап на долю выбросов).
"""

from __future__ import annotations

import numpy as np


def _robust_scale(y):
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    scale = 1.4826 * mad + 1e-12
    return med, scale


def detect_outliers(y, dy, *, threshold=5.0, max_fraction=0.10):
    """Выбросы по MAD: |y-med|/(1.4826·MAD) > threshold.

    Дополнительное условие: отклонение должно превышать 3-кратную локальную
    ошибку измерения (иначе флагом «выброс» может стать просто шумная точка).
    Число выбросов ограничено max_fraction·N (защита от абсурда).
    """
    y = np.asarray(y, float)
    dy = np.asarray(dy, float)
    n = len(y)
    med, scale = _robust_scale(y)
    z = np.abs(y - med) / scale
    outlier_mask = z > threshold
    noise_ok = np.abs(y - med) > 3.0 * dy
    outlier_mask &= noise_ok
    # кап по доле
    cap = int(max(1, max_fraction * n))
    if outlier_mask.sum() > cap:
        order = np.argsort(z)[::-1][:cap]
        outlier_mask = np.zeros(n, bool)
        outlier_mask[order] = True
    idx = np.where(outlier_mask)[0]
    return {
        "count": int(len(idx)),
        "indices": [int(i) for i in idx],
        "fraction": float(len(idx) / n),
        "median": float(med),
        "mad_scale": float(scale),
    }


def _transient_runs(t, y, dy, *, significance, min_points, direction):
    """Вспышки (direction=+1) или провалы (direction=-1).

    Базовый уровень — медиана с исключением уже найденных экстремумов;
    значимость — (y - base)/(шумовая оценка) с учётом dy.
    """
    y = np.asarray(y, float)
    dy = np.asarray(dy, float)
    n = len(y)
    med, scale = _robust_scale(y)
    # шумовая оценка на точку: масштаб + ошибка измерения
    noise = np.sqrt(scale ** 2 + dy ** 2)
    dev = (y - med) * direction
    sig = dev / np.maximum(noise, 1e-12)
    mask = sig > significance

    runs, cur = [], []
    for i in range(n):
        if mask[i]:
            cur.append(i)
        else:
            if len(cur) >= min_points:
                runs.append(cur)
            cur = []
    if len(cur) >= min_points:
        runs.append(cur)

    result = []
    for run in runs:
        peak_i = int(run[np.argmax(sig[run])])
        amp = float((y[peak_i] - med) * direction)
        result.append({
            "start": int(run[0]),
            "end": int(run[-1]),
            "n_points": len(run),
            "peak_index": peak_i,
            "peak_time_days": float(t[peak_i]),
            "amplitude": amp,
            "significance": float(sig[peak_i]),
        })
    return result


def detect_flares(t, y, dy, *, significance=5.0, min_points=3):
    runs = _transient_runs(t, y, dy, significance=significance,
                           min_points=min_points, direction=1)
    return {"count": len(runs), "flares": runs}


def detect_dips(t, y, dy, *, significance=4.0, min_points=4):
    runs = _transient_runs(t, y, dy, significance=significance,
                           min_points=min_points, direction=-1)
    return {"count": len(runs), "dips": runs}


def series_stats(t, y, dy):
    """Базовые статистики ряда и покрытия (пропуски, джиттер каденса)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    dy = np.asarray(dy, float)
    n = len(t)
    span = float(np.ptp(t))
    dt = np.diff(t)
    med, scale = _robust_scale(y)
    std = float(np.std(y))
    return {
        "n_points": n,
        "t_start_days": float(t[0]),
        "t_end_days": float(t[-1]),
        "span_days": span,
        "median_dt_days": float(np.median(dt)),
        "min_dt_days": float(np.min(dt)),
        "max_gap_days": float(np.max(dt)),
        "n_gaps_gt_5d": int((dt > 5.0).sum()),
        "coverage_fraction": float(n * float(np.median(dt)) / max(span, 1e-9)),
        "mean_flux": float(np.mean(y)),
        "std_flux": std,
        "median_flux": float(med),
        "mad_scale": float(scale),
        "cv": float(std / max(abs(float(np.mean(y))), 1e-9)),
        "mean_err": float(np.mean(dy)),
    }