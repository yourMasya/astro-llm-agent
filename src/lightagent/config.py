"""Загрузка конфигурации проекта (config.yaml) в плоский доступ."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict
    root: pathlib.Path

    def section(self, name: str) -> dict:
        return self.raw.get(name, {})

    def get(self, *path: str, default=None):
        cur = self.raw
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur


def load_config(path: str | pathlib.Path | None = None) -> Config:
    cfg_path = pathlib.Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=cfg_path.parent)