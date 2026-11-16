"""Обёртка над ollama: суть голосового + вариант ответа.

Наружу торчат ровно две вещи: summarize() и исключение LLMUnavailable.
Хендлер ловит LLMUnavailable и показывает транскрипт без summary —
текст распознавания при падении LLM не теряется.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass

import ollama

log = logging.getLogger(__name__)

MODEL = os.getenv("OLLAMA_MODEL", "saiga")
HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))
MAX_INPUT_CHARS = int(os.getenv("LLM_MAX_INPUT_CHARS", "6000"))

SYSTEM_PROMPT = (
    "Ты разбираешь расшифровки голосовых сообщений на русском языке.\n"
    "Отвечай ТОЛЬКО на русском и СТРОГО в таком формате, ровно две строки:\n"
    "СУТЬ: <о чём сообщение, 1-2 предложения>\n"
    "ОТВЕТ: <готовый вариант ответа собеседнику от первого лица, 1-3 предложения>\n\n"
    "Запрещено: markdown, списки, заголовки, пояснения, извинения, любой текст "
    "до строки СУТЬ: и после строки ОТВЕТ:. Не выдумывай факты, которых нет в расшифровке. "
    "Если сообщение бессвязное — так и напиши в строке СУТЬ."
)

# Допускаем: markdown-звёздочки, «СУТЬ»/«Суть», двоеточие/тире, перевод строки после метки.
_SUMMARY_RE = re.compile(
    r"(?:^|\n)[*_\s>#-]*сут[ьи]\s*[*_]*\s*[:\-–—]\s*(.+?)(?=\n[*_\s>#-]*ответ\s*[*_]*\s*[:\-–—]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_REPLY_RE = re.compile(
    r"(?:^|\n)[*_\s>#-]*ответ\s*[*_]*\s*[:\-–—]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


class LLMUnavailable(Exception):
    """Ollama не ответила: не запущена, нет модели, таймаут, сетевая ошибка."""


@dataclass(slots=True)
class Summary:
    summary: str
    reply: str
    strict_format: bool  # False — формат не соблюдён, собрано фолбэком


def _clean(value: str) -> str:
    value = value.strip().strip("`").strip()
    value = re.sub(r"\s*\n\s*", " ", value)      # ответ в несколько строк -> одна
    value = re.sub(r"^[*_\s>#-]+", "", value)
    value = re.sub(r"[*_]+", "", value)
    return value.strip()


def _first_sentences(text: str, limit: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    joined = " ".join(s for s in sentences[:limit] if s).strip()
    return joined or text.strip()


def parse_response(raw: str) -> Summary:
    """Разобрать ответ LLM. Никогда не бросает — при любом мусоре даёт фолбэк."""
    raw = (raw or "").strip()
    if not raw:
        return Summary(summary="", reply="", strict_format=False)

    summary_match = _SUMMARY_RE.search(raw)
    reply_match = _REPLY_RE.search(raw)

    summary = _clean(summary_match.group(1)) if summary_match else ""
    reply = _clean(reply_match.group(1)) if reply_match else ""

    if summary and reply:
        return Summary(summary=summary, reply=reply, strict_format=True)

    # Фолбэк: модель проигнорировала формат. Вытаскиваем хоть что-то полезное.
    body = _clean(raw)
    if not summary:
        summary = _first_sentences(body)
    if not reply:
        # если после сути осталось ещё что-то — считаем это вариантом ответа
        tail = body[len(summary):].strip() if body.startswith(summary) else ""
        reply = _first_sentences(tail, limit=2) if tail else ""
    return Summary(summary=summary, reply=reply, strict_format=False)


def _extract_content(response: object) -> str:
    """ollama-python отдаёт ChatResponse (новые версии) или dict (старые)."""
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            return str(content)
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))
    return ""


async def summarize(transcript: str) -> Summary:
    """Суть + вариант ответа. Бросает LLMUnavailable, если ollama недоступна."""
    text = transcript.strip()
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + " […обрезано]"

    client = ollama.AsyncClient(host=HOST)
    try:
        response = await asyncio.wait_for(
            client.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Расшифровка голосового:\n{text}"},
                ],
                options={"temperature": 0.3, "num_predict": 400},
            ),
            timeout=TIMEOUT,
        )
    except asyncio.TimeoutError as exc:
        raise LLMUnavailable(f"ollama не ответила за {TIMEOUT:.0f} с") from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # ResponseError, ConnectError, httpx-ошибки и т.п.
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc

    content = _extract_content(response)
    if not content.strip():
        raise LLMUnavailable("пустой ответ от ollama")

    result = parse_response(content)
    if not result.strict_format:
        log.warning("LLM ответила не в формате СУТЬ:/ОТВЕТ:, использую фолбэк: %r", content[:200])
    return result
