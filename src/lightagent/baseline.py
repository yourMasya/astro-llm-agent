"""Детерминированный baseline: фиксированный конвейер без LLM.

Baseline — это «честная» версия того же набора численных методов, но с
жёстким порядком вызовов и фиксированными порогами из config.yaml. Он не
понимает естественный язык: единицы всегда сутки, вызываются все инструменты
подряд (дороже по вызовам, чем целевой агент) и ответ не адаптируется к
вопросу. Сравнение с ним показывает ценность LLM-планирования при той же
численной базе.

Вызовы идут через тот же ToolExecutor, что и у агента, и пишутся в логи —
стоимость и надёжность сравнимы по одной шкале. LLM-токенов baseline не
тратит (cost = 0).
"""

from __future__ import annotations

import pathlib
import time

from .config import Config
from .data.loader import Dataset
from .agent.executor import ToolExecutor


class BaselineRunner:
    def __init__(self, ds: Dataset, cfg: Config, *, log_dir=None,
                 harmonic_check=True, alias_check=True):
        self.ds = ds
        self.cfg = cfg
        exec_cfg = cfg.section("agent")
        self.executor = ToolExecutor(
            ds, cfg, obs_char_limit=exec_cfg.get("obs_char_limit", 3500))
        self.log_dir = pathlib.Path(log_dir) if log_dir else \
            cfg.root / "logs" / "baseline"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.harmonic_check = harmonic_check
        self.alias_check = alias_check
        self.stats = {"tool_calls": 0, "retries_total": 0, "errors": 0,
                      "tool_names": []}

    def _call(self, name, args) -> dict:
        t0 = time.monotonic()
        res = self.executor.call(name, args)
        self.stats["tool_calls"] += 1
        self.stats["retries_total"] += res.retries
        if not res.ok:
            self.stats["errors"] += 1
        if res.name not in self.stats["tool_names"]:
            self.stats["tool_names"].append(res.name)
        import json as _json
        with open(self.log_dir / f"{self.task_id}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({
                "task_id": self.task_id, "kind": "tool", "tool": res.name,
                "args": args, "status": res.status, "error": res.error,
                "retries": res.retries,
                "duration_ms": round(res.duration_ms, 2),
                "text": res.text[:800],
            }, ensure_ascii=False) + "\n")
        return res

    def run(self, task: dict) -> dict:
        """Детерминированная обработка задачи: полный конвейер, сутки."""
        self.task_id = task["id"]
        curve_id = task["curve_id"]
        a = self.cfg.section("analysis")
        fap_thr = float(a["fap_threshold"])
        notes = []

        info = self._call("get_curve_info", {"curve_id": curve_id})
        period = self._call("periodogram", {
            "curve_id": curve_id, "time_unit": "days",
            "samples_per_peak": a["samples_per_peak"],
            "n_peaks": a["n_peaks"]})
        outliers = self._call("detect_outliers", {
            "curve_id": curve_id,
            "threshold": a["outlier"]["threshold"]})
        flares = self._call("detect_flares", {
            "curve_id": curve_id,
            "significance": a["flare"]["significance"],
            "min_points": a["flare"]["min_points"]})
        dips = self._call("detect_dips", {
            "curve_id": curve_id,
            "significance": a["dip"]["significance"],
            "min_points": a["dip"]["min_points"]})

        periodic, period_days = None, None
        if period.ok and period.data.get("peaks"):
            peaks = period.data["peaks"]
            sig = [p for p in peaks if p.get("significant")]
            if sig:
                cand = sig[0]
                period_days = cand["period_days"]
                periodic = True
                if self.harmonic_check and cand.get("subharmonic_similar"):
                    period_days = 2.0 * cand["period_days"]
                    notes.append("субгармоника: фундаментальный период удвоен "
                                 "(два провала за цикл, затменная двойная)")
                if self.alias_check and cand.get("one_day_alias_note"):
                    notes.append(cand["one_day_alias_note"])
            else:
                periodic = False
                notes.append("значимых пиков периодограммы нет (fap > "
                             f"{fap_thr:.0e}); периодичность не установлена")
        elif period.ok:
            notes.append("периодограмма не дала пиков")

        if info.ok:
            inf = info.data
            if inf.get("coverage_fraction", 1.0) < 0.5:
                notes.append(f"низкое покрытие ({inf['coverage_fraction']:.2f}): "
                             "выводы менее надёжны")
            if inf.get("n_gaps_gt_5d", 0) > 2:
                notes.append(f"много пропусков >5 сут: {inf['n_gaps_gt_5d']}")

        final = {
            "final": True,
            "periodic": periodic,
            "period_days": period_days,
            "period_unit": "days",
            "outliers_count": outliers.data["count"] if outliers.ok else None,
            "outliers_indices": outliers.data["indices"] if outliers.ok else [],
            "flares_count": flares.data["count"] if flares.ok else None,
            "dips_count": dips.data["count"] if dips.ok else None,
            "notes": "; ".join(notes) if notes else "детерминированный конвейер",
        }
        stats = {
            "task_id": task["id"], "steps": None, "tool_calls":
            self.stats["tool_calls"], "llm_calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "est_cost_usd": 0.0,
            "retries_total": self.stats["retries_total"], "degraded": False,
            "tool_names": list(self.stats["tool_names"]),
            "duration_s": 0.0, "final": final,
        }
        return {"final": final, "stats": stats}