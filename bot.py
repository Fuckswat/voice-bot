"""Точка входа: голосовые -> whisper -> saiga -> суть + вариант ответа."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # до импорта сервисов: они читают конфиг из окружения на импорте

from aiogram import Bot, Dispatcher  # noqa: E402

from handlers.voice import router as voice_router  # noqa: E402
from services import stt  # noqa: E402
from services.storage import storage  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")


async def _notify_startup(bot: Bot) -> None:
    """Пингануть владельца при старте, если в .env задан NOTIFY_CHAT_ID."""
    chat_id = os.getenv("NOTIFY_CHAT_ID", "").strip()
    if not chat_id:
        return
    device, compute_type = stt._pick_device()
    try:
        await bot.send_message(
            chat_id,
            f"✅ бот поднялся\nwhisper: {stt.MODEL_SIZE} ({device}/{compute_type})\n"
            f"ollama: {os.getenv('OLLAMA_MODEL', 'saiga')}\n\nкидай гс.",
        )
    except Exception:
        log.warning("не смог отправить стартовое уведомление в чат %s", chat_id, exc_info=True)


async def main() -> None:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        sys.exit("BOT_TOKEN не задан. Скопируй .env.example в .env и впиши токен от @BotFather.")

    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(voice_router)

    await storage.connect()
    removed = await storage.purge_expired()
    if removed:
        log.info("на старте удалено протухших транскриптов: %d", removed)

    cleanup_task = asyncio.create_task(storage.cleanup_loop(), name="ttl-cleanup")

    warmup_task: asyncio.Task | None = None
    if os.getenv("WHISPER_WARMUP", "1") != "0":
        # грузим веса параллельно поллингу: бот сразу живой, первое гс подождёт загрузку
        warmup_task = asyncio.create_task(stt.warmup(), name="whisper-warmup")

    try:
        me = await bot.get_me()
        log.info("бот @%s запущен", me.username)
        await bot.delete_webhook(drop_pending_updates=True)
        await _notify_startup(bot)
        await dp.start_polling(bot)
    finally:
        for task in (cleanup_task, warmup_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(t for t in (cleanup_task, warmup_task) if t is not None),
            return_exceptions=True,
        )
        await storage.close()
        await bot.session.close()
        log.info("остановлен")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
