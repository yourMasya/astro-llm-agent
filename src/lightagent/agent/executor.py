"""Исполнитель вызовов инструментов: валидация, повторные попытки, логи.

Поток выполнения одного вызова:
1. локальная валидация схемы (ToolValidationError) — LLM не может обойти;
2. семантическая проверка инструментом (ToolSemanticError) — понятная ошибка;
3. выполнение обработчика с повторными попытками на транзиентные ошибки;
4. сериализация результата (обрезка по лимиту контекста) и запись в лог.

Для стресс-тестов предусмотрена инъекция сбоев (inject_failure): инструмент
можен «падать» заданное число раз заданным исключением.
"""

from __future__ import annotations

import json
import logging
import time

from ..config import Config
from ..data.loader import Dataset
from .tools import (ToolRegistry, ToolSemanticError, ToolValidationError)

log = logging.getLogger("lightagent.executor")

TRANSIENT = (TimeoutError, ConnectionError, OSError)


class ToolResult:
    __slots__ = ("name", "ok", "status", "text", "data", "error", "retries",
                 "duration_ms", "args")

    def __init__(self, name, ok, status, text="", data=None, error=None,
                 retries=0, duration_ms=0.0, args=None):
        self.name = name
        self.ok = ok
        self.status = status  # ok | validation_error | semantic_error | error
        self.text = text
        self.data = data      # разобранный dict-результат (только при ok)
        self.error = error
        self.retries = retries
        self.duration_ms = duration_ms
        self.args = args or {}

    def to_dict(self):
        return {
            "name": self.name, "ok": self.ok, "status": self.status,
            "error": self.error, "retries": self.retries,
            "duration_ms": round(self.duration_ms, 2), "args": self.args,
        }


class ToolExecutor:
    def __init__(self, ds: Dataset, cfg: Config, *, obs_char_limit=3500,
                 retries_transient=2, retry_backoff_s=(0.5, 1.0)):
        self.registry = ToolRegistry(ds, cfg)
        self.cfg = cfg
        self.obs_char_limit = obs_char_limit
        self.retries_transient = retries_transient
        self.retry_backoff_s = retry_backoff_s
        self._injected = {}  # name -> (exception_factory, attempts_left)

    def inject_failure(self, name: str, exc_factory, attempts: int = 1):
        """Стресс-инъекция: инструмент падает `attempts` раз, далее работает."""
        self._injected[name] = (exc_factory, attempts)
        log.warning("injected failure for tool %s: %d attempts", name, attempts)

    def clear_failures(self):
        self._injected.clear()

    def _maybe_inject(self, name: str):
        entry = self._injected.get(name)
        if not entry:
            return
        factory, attempts = entry
        if attempts <= 0:
            self._injected.pop(name)
            return
        self._injected[name] = (factory, attempts - 1)
        exc = factory()
        raise exc

    def serialize(self, result: dict) -> str:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str, indent=1)
        except (TypeError, ValueError):
            text = str(result)
        return text[: self.obs_char_limit]

    def call(self, name: str, args: dict) -> ToolResult:
        """Полный цикл вызова инструмента с валидацией и ретраями."""
        t0 = time.monotonic()
        try:
            self.registry.validate_call(name, args)
        except ToolValidationError as exc:
            dur = (time.monotonic() - t0) * 1000
            res = ToolResult(name, ok=False, status="validation_error",
                             error=str(exc), duration_ms=dur, args=args)
            res.text = f"ошибка валидации: {exc}"
            return res

        retries = 0
        while True:
            try:
                self._maybe_inject(name)
                result = self.registry.run(name, args)
                dur = (time.monotonic() - t0) * 1000
                res = ToolResult(name, ok=True, status="ok", data=result,
                                 retries=retries, duration_ms=dur, args=args)
                res.text = self.serialize(result)
                return res
            except ToolSemanticError as exc:
                # семантическая ошибка: повторять бессмысленно
                dur = (time.monotonic() - t0) * 1000
                res = ToolResult(name, ok=False, status="semantic_error",
                                 error=str(exc), retries=retries,
                                 duration_ms=dur, args=args)
                res.text = f"ошибка инструмента: {exc}"
                return res
            except TRANSIENT as exc:
                if retries < self.retries_transient:
                    retries += 1
                    delay = self.retry_backoff_s[min(retries - 1,
                                                     len(self.retry_backoff_s) - 1)]
                    log.warning("tool %s transient failure %r, retry %d in %.1fs",
                                name, exc, retries, delay)
                    time.sleep(delay)
                    continue
                dur = (time.monotonic() - t0) * 1000
                res = ToolResult(name, ok=False, status="error",
                                 error=f"{type(exc).__name__}: {exc}",
                                 retries=retries, duration_ms=dur, args=args)
                res.text = (f"инструмент недоступен после {retries} повторных "
                            f"попыток ({type(exc).__name__}); попробуй другой "
                            f"инструмент или проверь аргументы")
                return res
            except Exception as exc:  # неизвестная ошибка: не повторяем
                dur = (time.monotonic() - t0) * 1000
                res = ToolResult(name, ok=False, status="error",
                                 error=f"{type(exc).__name__}: {exc}",
                                 retries=retries, duration_ms=dur, args=args)
                res.text = f"неожиданная ошибка инструмента {type(exc).__name__}: {exc}"
                return res

    def observation_text(self, res: ToolResult) -> str:
        """Текст наблюдения для планировщика."""
        status_ru = {"ok": "успешно", "validation_error": "ошибка валидации",
                     "semantic_error": "семантическая ошибка", "error": "ошибка"}
        head = (f"Инструмент {res.name} → {status_ru.get(res.status, res.status)}"
                f" за {res.duration_ms:.0f} мс")
        if res.retries:
            head += f" (повторных попыток: {res.retries})"
        return head + ":\n" + res.text