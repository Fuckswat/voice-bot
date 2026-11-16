"""Хендлеры: приём голосовых и выдача полного транскрипта по кнопке."""

from __future__ import annotations

import html
import logging
import os
import tempfile
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from services import llm, stt
from services.storage import storage

log = logging.getLogger(__name__)
router = Router(name="voice")

TMP_DIR = Path(os.getenv("TMP_DIR", tempfile.gettempdir())) / "voice-bot"
TMP_DIR.mkdir(parents=True, exist_ok=True)

LONG_VOICE_SECONDS = int(os.getenv("LONG_VOICE_SECONDS", "120"))
MAX_FILE_BYTES = 20 * 1024 * 1024  # лимит getFile у Bot API
TELEGRAM_LIMIT = 4096
CHUNK = 3900

CB_PREFIX = "full:"


def _fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _quality_emoji(confidence: int) -> str:
    if confidence > 70:
        return "🟢"
    if confidence >= 40:
        return "🟡"
    return "🔴"


def _keyboard(message_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📄 полный текст", callback_data=f"{CB_PREFIX}{message_id}")]
        ]
    )


def _chunks(text: str, size: int = CHUNK) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


async def _safe_edit(message: Message, text: str, **kwargs) -> None:
    """Правка сообщения не должна ронять обработку (message is not modified и т.п.)."""
    try:
        await message.edit_text(text, **kwargs)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return
        log.warning("не смог отредактировать сообщение: %s", exc)
    except Exception:
        log.exception("не смог отредактировать сообщение")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    log.info("/start от chat_id=%s (@%s)", message.chat.id,
             message.from_user.username if message.from_user else "?")
    await message.answer(
        "Привет. Кидай голосовое — распознаю, скажу суть и предложу вариант ответа.\n\n"
        "🟢 / 🟡 / 🔴 — насколько уверенно распозналась речь.\n"
        "Полный текст лежит под кнопкой «📄 полный текст» 24 часа."
    )


@router.message(F.voice)
async def handle_voice(message: Message, bot: Bot) -> None:
    voice = message.voice
    if voice is None:  # для type-checker'а: фильтр F.voice это уже гарантировал
        return

    if voice.file_size and voice.file_size > MAX_FILE_BYTES:
        await message.reply(
            "🚫 гс слишком большое — Telegram не отдаёт ботам файлы больше 20 МБ."
        )
        return

    is_long = voice.duration >= LONG_VOICE_SECONDS
    status_text = "🎙 слушаю гс..."
    if is_long:
        status_text += f"\n⏳ длинное ({_fmt_duration(voice.duration)}) — обработка займёт время"
    status = await message.reply(status_text)

    tmp_path = TMP_DIR / f"{message.chat.id}_{message.message_id}.ogg"
    try:
        await bot.download(voice, destination=tmp_path)
        await _safe_edit(status, "⏳ распознаю речь...")

        result = await stt.transcribe(str(tmp_path))

        if result.is_empty:
            await _safe_edit(
                status,
                "🔇 не разобрал речь — похоже, там тишина или шум. Попробуй записать заново.",
            )
            return

        # Сохраняем ДО похода в LLM: даже если ollama упадёт, транскрипт не потеряется.
        await storage.save(
            chat_id=message.chat.id,
            message_id=message.message_id,
            user_id=message.from_user.id if message.from_user else None,
            text=result.text,
            confidence=result.confidence,
            duration=result.duration,
        )

        await _safe_edit(status, "🧠 думаю над сутью и ответом...")

        emoji = _quality_emoji(result.confidence)
        header = (
            f"{emoji} <b>распознано на {result.confidence}%</b> "
            f"· {_fmt_duration(result.duration)}"
        )

        try:
            summary = await llm.summarize(result.text)
        except llm.LLMUnavailable as exc:
            log.warning("ollama недоступна: %s", exc)
            preview = html.escape(result.text)
            body = (
                "\n\n⚠️ <i>LLM недоступна — суть и вариант ответа не собрал, "
                "вот распознанный текст:</i>\n\n"
                f"{preview}"
            )
            text = header + body
        else:
            if not summary.summary and not summary.reply:
                # LLM ответила, но вытащить из ответа нечего — не показываем пустоту
                summary.summary = result.text[:500]
            parts = [header, ""]
            if summary.summary:
                parts.append(f"<b>суть:</b> {html.escape(summary.summary)}")
            if summary.reply:
                parts.append("")
                parts.append(f"<b>вариант ответа:</b> {html.escape(summary.reply)}")
            if not summary.strict_format:
                parts.append("")
                parts.append("<i>⚠️ модель ответила не по формату, разобрал как смог</i>")
            text = "\n".join(parts)

        if len(text) > TELEGRAM_LIMIT:
            text = text[: TELEGRAM_LIMIT - 20].rstrip() + "\n…"

        await _safe_edit(
            status,
            text,
            parse_mode="HTML",
            reply_markup=_keyboard(message.message_id),
        )

    except Exception:
        log.exception("обработка гс упала (chat=%s, msg=%s)", message.chat.id, message.message_id)
        await _safe_edit(
            status,
            "💥 не смог обработать это гс. Попробуй ещё раз — если повторится, "
            "скорее всего проблема на стороне бота.",
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            log.warning("не удалось удалить временный файл %s", tmp_path, exc_info=True)


@router.callback_query(F.data.startswith(CB_PREFIX))
async def send_full_transcript(callback: CallbackQuery, bot: Bot) -> None:
    # callback.message может быть InaccessibleMessage (у него нет .answer()),
    # поэтому отвечаем через bot.send_message по chat_id.
    target = callback.message
    if target is None:
        await callback.answer("сообщение недоступно", show_alert=True)
        return
    chat_id = target.chat.id

    raw_id = (callback.data or "")[len(CB_PREFIX) :]
    try:
        voice_message_id = int(raw_id)
    except ValueError:
        await callback.answer("кнопка испорчена", show_alert=True)
        return

    try:
        record = await storage.get(chat_id, voice_message_id)
    except Exception:
        log.exception("не смог достать транскрипт из sqlite")
        await callback.answer("не смог достать текст из базы, попробуй позже", show_alert=True)
        return

    if record is None:
        await callback.answer(
            "Полный текст уже недоступен: он хранится 24 часа. Перешли гс заново.",
            show_alert=True,
        )
        return

    await callback.answer()
    chunks = _chunks(record.text)
    for index, chunk in enumerate(chunks):
        prefix = "📄 <b>полный текст:</b>\n\n" if index == 0 else ""
        await bot.send_message(chat_id, prefix + html.escape(chunk), parse_mode="HTML")
