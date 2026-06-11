from __future__ import annotations

import os
from typing import Any, Dict, List

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_TEMPERATURE = 0.15
DEFAULT_TIMEOUT = 120


def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    Call local vLLM OpenAI-compatible Chat Completions API and return assistant text only.
    """
    base_url = os.getenv("VLLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.getenv("VLLM_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    timeout = int(os.getenv("VLLM_TIMEOUT_SEC", str(DEFAULT_TIMEOUT)))

    url = f"{base_url}/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()
    except requests.RequestException as exc:
        raise RuntimeError(f"vLLM request failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"vLLM response parse failed: {exc}") from exc
