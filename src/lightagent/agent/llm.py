"""Клиент LLM (OpenAI-совместимый endpoint, корпоративный LiteLLM-прокси).

Используется только как планировщик: модель возвращает JSON-действия, а числа
появляются исключительно в результатах инструментов. Стоимость оценивается по
тарифам deepseek-chat (config.yaml), так как точная цена прокси неизвестна.
"""

from __future__ import annotations

import json
import os
import time

import requests


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, cfg, *, model=None, retries=6, backoff_s=(2, 5, 10, 20, 30, 30)):
        lc = cfg.section("llm")
        self.base_url = lc.get("base_url", "https://litellm-proxy.ai.yadro.com").rstrip("/")
        self.api_key = os.environ.get(lc.get("api_key_env", "INFERENCE_API_KEY"), "")
        if not self.api_key:
            raise LLMError(f"переменная окружения {lc['api_key_env']} не задана")
        self.model = model or lc.get("model", "DeepSeek-V4-Flash")
        self.temperature = float(lc.get("temperature", 0.0))
        self.max_tokens = int(lc.get("max_tokens_per_step", 2000))
        self.retries = retries
        self.backoff_s = backoff_s
        self.price_input = float(lc.get("price_input_per_mtok", 0.0))
        self.price_output = float(lc.get("price_output_per_mtok", 0.0))

    def chat(self, messages, *, max_tokens=None) -> tuple[str, dict]:
        """Один вызов chat completion. Возвращает (текст, usage_meta)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                meta = {
                    "model": self.model,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "attempts": attempt,
                }
                if not meta["prompt_tokens"]:
                    meta["prompt_tokens"] = max(1, sum(len(m.get("content", ""))
                                                       for m in messages) // 4)
                if not meta["completion_tokens"]:
                    meta["completion_tokens"] = max(1, len(text) // 4)
                return text, meta
            except (requests.RequestException, LLMError, KeyError, ValueError) as exc:
                last_err = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_s[min(attempt - 1, len(self.backoff_s) - 1)])
        raise LLMError(f"LLM недоступен после {self.retries} попыток: {last_err}")

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1e6 * self.price_input
                + completion_tokens / 1e6 * self.price_output)

    @staticmethod
    def render_action(action: dict) -> str:
        """Компактное представление действия для логов."""
        return json.dumps(action, ensure_ascii=False)[:500]