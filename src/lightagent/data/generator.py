"""Генератор бенчмарк-данных: реалистичные кривые блеска с ground truth.

Моделируются свойства наземных фотометрических обзоров (OGLE/ASAS-SN):
неравномерная дискретизация с джиттером, сезонные окна с пропусками,
случайные пропуски точек, гетероскедастичный шум. Сигналы — типичные
астрофизические: периодические (RR-лироподобные/цефеидоподобные),
затменные двойные (провалы), вспышки, выбросы, тренд, стационарный шум.

Почему синтетика с известной истиной: задание требует «ожидаемых проверяемых
результатов» по 20 задачам, включая позиции аномалий. У открытых данных
(OGLE/ASAS-SN) нет попиксельной разметки аномалий, поэтому количественную
точность можно измерить только на контролируемом наборе. Загрузчик
реальных CSV предусмотрен в loader.py; формат тот же.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

import numpy as np

CLASSES = ["periodic", "eclipsing", "flare", "quiet", "outliers", "trend"]


@dataclass
class Curve:
    id: str
    t: np.ndarray        # время, сутки (относительные, MJD-подобные)
    flux: np.ndarray     # относительный поток, средний ~1
    flux_err: np.ndarray # 1-сигма ошибка измерения (гетероскедастична)


def _rng_for(curve_id: str) -> np.random.Generator:
    """Детерминированный ГПСЧ по id кривой."""
    seed = int(hashlib.sha256(curve_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _observation_times(rng, n, *, seasons=2, season_len_d=120.0, gap_d=250.0,
                       cadence_base_d=1.0, cadence_jitter=(0.3, 1.7),
                       drop_fraction=0.10):
    """Неравномерные моменты наблюдения: сезонные окна + джиттер + пропуски."""
    per_season = max(3, n // seasons)
    blocks = []
    t0 = 0.0
    made = 0
    while made < n:
        n_s = min(per_season, n - made)
        # джиттер каденса в пределах сезона
        dts = rng.uniform(*cadence_jitter, size=n_s) * cadence_base_d
        t = t0 + np.cumsum(dts)
        # обрезаем, чтобы не вылезать за сезонную границу (имитация погоды)
        overflow = t > t0 + season_len_d
        if overflow.any():
            t = t[t <= t0 + season_len_d]
        if len(t) == 0:
            t0 += season_len_d + gap_d
            continue
        blocks.append(t)
        made += len(t)
        t0 += season_len_d + gap_d
    t = np.concatenate(blocks)[:n]
    # случайные пропуски (потерянные наблюдения)
    if isinstance(drop_fraction, (list, tuple)):
        drop_fraction = float(rng.uniform(*drop_fraction))
    keep = rng.random(len(t)) >= drop_fraction
    t = np.sort(t[keep])
    return t


def _base_model(rng, curve_type: str, t: np.ndarray, *, sigma_base=0.016,
                outlier_amp_sigma=None, flare_amp_sigma=None,
                dip_depth_sigma=None):
    """Истинный сигнал модели + ground truth аномалий (по индексам).

    Амплитуды аномалий: по умолчанию — абсолютные единицы (v1, вырожденный
    режим); если заданы *_amp_sigma-кортежи (lo, hi), амплитуда берется как
    U(lo,hi)·sigma_base — реалистичный режим сигнал/шум (протокол v2).
    """
    n = len(t)
    flux_model = np.ones(n)
    truth = {}

    if curve_type in ("periodic",):
        p_true = rng.uniform(2.5, 25.0)
        a = rng.uniform(0.04, 0.25)
        a2 = rng.uniform(0.2, 0.6)
        a3 = rng.uniform(0.1, 0.4)
        phase0 = rng.uniform(0, 2 * np.pi)
        phi = 2 * np.pi * (t - t[0]) / p_true + phase0
        flux_model += a * (np.sin(phi) + a2 * np.sin(2 * phi) + a3 * np.sin(3 * phi))
        truth.update(P_true=float(p_true), amplitude=float(a),
                     harmonics=[float(a2), float(a3)])

    elif curve_type == "eclipsing":
        # Провалы в ДНЯХ (не в долях периода): при каденсе ~1 сут иначе
        # в провал попадает 0-1 точка. Вторичный провал почти на порядок
        # мельче первичного: иначе периодограмма «видит» два одинаковых
        # провала за цикл и ошибочно даёт P/2 — типичная ловушка затменных.
        p_true = rng.uniform(14.0, 30.0)
        if dip_depth_sigma is not None:
            lo, hi = dip_depth_sigma
            depth1 = float(rng.uniform(lo, hi) * sigma_base)
            depth2 = float(rng.uniform(1.2, 2.5) * sigma_base)
            truth["dip_depth_sigma"] = [lo, hi]
        else:
            depth1 = rng.uniform(0.08, 0.25)
            depth2 = rng.uniform(0.010, 0.020)
        w1 = rng.uniform(2.0, 3.5)      # сутки
        w2 = rng.uniform(1.0, 2.0)
        phase0 = rng.uniform(0, p_true)
        phase = ((t - t[0] + phase0) % p_true)
        dip1 = phase < w1
        dip2 = (phase > p_true / 2) & (phase < p_true / 2 + w2)
        flux_model[dip1] -= depth1
        flux_model[dip2] -= depth2
        truth.update(P_true=float(p_true), depth1=float(depth1),
                     depth2=float(depth2))
        runs = _runs_of(np.where(dip1 | dip2)[0], n)
        truth["dips"] = [[int(a), int(b)] for a, b in runs]

    elif curve_type == "flare":
        # Вспышка = контур гауссианы; ground truth — СМЕЖНЫЕ группы индексов,
        # где модельный поток поднят выше 2·sigma_base (выше уровня шума).
        n_flares = int(rng.integers(2, 5))
        elevation = np.zeros(n)
        for _ in range(n_flares):
            if flare_amp_sigma is not None:
                # sigma-режим: привязываем вспышку к реальному моменту
                # наблюдения. Равномерный выбор tc по всему span часто попадает
                # в сезонный гэп (без точек) — вспышка «теряется» в данных.
                i0 = int(rng.integers(5, n - 5))
                tc = float(t[i0]) + float(rng.uniform(-0.4, 0.4))
                w = rng.uniform(1.5, 3.5)
                lo, hi = flare_amp_sigma
                amp = float(rng.uniform(lo, hi) * sigma_base)
                truth.setdefault("flare_amp_sigma", [lo, hi])
            else:
                tc = rng.uniform(t[0] + 8, t[-1] - 8)
                w = rng.uniform(1.5, 3.5)
                amp = rng.uniform(0.15, 0.50)
            mask = (t > tc - 3 * w) & (t < tc + 3 * w)
            elevation[mask] += amp * np.exp(-((t[mask] - tc) / w) ** 2)
        flux_model += elevation
        flare_mask = elevation > 2.0 * sigma_base
        truth["flares"] = [[int(a), int(b)]
                           for a, b in _runs_of(np.where(flare_mask)[0], n)]

    elif curve_type == "outliers":
        k = max(3, int(round(rng.uniform(0.02, 0.05) * n)))
        idx = rng.choice(n, size=k, replace=False)
        signs = np.where(rng.random(k) < 0.7, 1.0, -1.0)
        if outlier_amp_sigma is not None:
            lo, hi = outlier_amp_sigma
            flux_model[idx] += signs * rng.uniform(lo, hi, size=k) * sigma_base
            truth["outlier_amp_sigma"] = [lo, hi]
        else:
            flux_model[idx] += signs * rng.uniform(4.0, 8.0, size=k)
        truth["outliers_indices"] = sorted(int(i) for i in idx)

    elif curve_type == "trend":
        slope = rng.uniform(0.0002, 0.0008)
        t_mid = 0.5 * (t[0] + t[-1])
        flux_model += slope * (t - t_mid)
        truth["slope"] = float(slope)

    # quiet: ничего, только шум
    truth["type"] = curve_type
    return flux_model, truth


def _runs_of(idx: np.ndarray, n: int):
    """Смежные группы индексов в [0, n)."""
    if len(idx) == 0:
        return []
    idx = np.sort(idx)
    starts = [idx[0]]
    ends = []
    prev = idx[0]
    delta = np.diff(idx)
    bounds = np.where(delta > 1)[0]
    # разбиение по границам разрывов
    segs = np.split(idx, bounds + 1)
    return [(int(s[0]), int(s[-1])) for s in segs]


def generate_curve(curve_id: str, curve_type: str | None = None,
                   **gen_kw) -> tuple[Curve, dict]:
    """Сгенерировать одну кривую. Возвращает (Curve, truth)."""
    rng = _rng_for(curve_id)
    if curve_type is None:
        curve_type = str(rng.choice(CLASSES))

    n = int(rng.integers(150, 400))
    obs_keys = ("seasons", "season_len_d", "gap_d", "cadence_base_d",
                "cadence_jitter", "drop_fraction")
    obs_kw = {k: v for k, v in gen_kw.items() if k in obs_keys}
    t = _observation_times(rng, n, **obs_kw)
    sigma_base = rng.uniform(gen_kw.get("noise_base", [0.008, 0.025])[0],
                             gen_kw.get("noise_base", [0.008, 0.025])[1])
    extra = {k: gen_kw[k] for k in ("outlier_amp_sigma", "flare_amp_sigma",
                                    "dip_depth_sigma") if k in gen_kw}
    flux_model, truth = _base_model(rng, curve_type, t, sigma_base=sigma_base,
                                    **extra)
    flux_err = sigma_base * (1.0 + 0.4 * rng.random(len(t)))
    flux = flux_model + rng.normal(0.0, flux_err)
    truth["n_points"] = int(len(t))
    truth["curve_id"] = curve_id
    truth.setdefault("P_true", None)
    truth.setdefault("outliers_indices", [])
    truth.setdefault("flares", [])
    truth.setdefault("dips", [])
    return Curve(id=curve_id, t=t, flux=flux, flux_err=flux_err), truth


def generate_dataset(n_curves: int, seed: int = 42, **gen_kw) -> dict:
    """Сгенерировать все кривые датасета (детерминированно по seed и id)."""
    rng = np.random.default_rng(seed)
    curves: dict[str, Curve] = {}
    truths: dict[str, dict] = {}
    for i in range(n_curves):
        cid = f"b{i:03d}"
        ctype = str(rng.choice(CLASSES))
        curve, truth = generate_curve(cid, ctype, **gen_kw)
        curves[cid] = curve
        truths[cid] = truth
    return curves, truths


def save_dataset(curves: dict[str, Curve], truths: dict[str, dict],
                 csv_path, truth_path, card_path):
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("id,time_days,flux,flux_err\n")
        for cid, c in curves.items():
            for t, f, e in zip(c.t, c.flux, c.flux_err):
                fh.write(f"{cid},{t:.6f},{f:.7f},{e:.7f}\n")
    with open(truth_path, "w", encoding="utf-8") as fh:
        json.dump(truths, fh, indent=2, ensure_ascii=False)
    _write_card(curves, truths, card_path)


def _write_card(curves, truths, card_path):
    lines = [
        "# Карточка набора данных (dataset card)",
        "",
        "Синтетический бенчмарк кривых блеска, калиброванный на свойства "
        "наземных фотометрических обзоров (OGLE/ASAS-SN): неравномерная "
        "дискретизация, сезонные пропуски, гетероскедастичный шум.",
        "",
        f"- кривых: **{len(curves)}**",
        "- поток в относительных единицах, средний ~1",
        "- время в сутках (относительные, MJD-подобные)",
        "- классы: " + ", ".join(CLASSES),
        "",
        "| id | класс | N точек | P_true, сут | выбросы | вспышки | провалы |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid in sorted(curves):
        tr = truths[cid]
        p = f"{tr['P_true']:.3f}" if tr.get("P_true") else "—"
        lines.append(
            f"| {cid} | {tr['type']} | {tr['n_points']} | {p} | "
            f"{len(tr.get('outliers_indices', []))} | "
            f"{len(tr.get('flares', []))} | {len(tr.get('dips', []))} |"
        )
    pathlib.Path(card_path).write_text("\n".join(lines) + "\n", encoding="utf-8")