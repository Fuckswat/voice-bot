"""SQLite-слой для транскриптов.

Хранит полный текст распознавания, чтобы кнопка «полный текст» работала
после рестарта бота. Ключ записи — (chat_id, message_id) исходного голосового.
Записи старше TTL удаляются фоновой задачей.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

log = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("DB_PATH", "transcripts.db"))
TTL_SECONDS = float(os.getenv("TRANSCRIPT_TTL_HOURS", "24")) * 3600
CLEANUP_INTERVAL = float(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    chat_id     INTEGER NOT NULL,
    message_id  INTEGER NOT NULL,
    user_id     INTEGER,
    text        TEXT    NOT NULL,
    confidence  INTEGER NOT NULL DEFAULT 0,
    duration    REAL    NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_transcripts_created_at ON transcripts (created_at);
"""


@dataclass(slots=True)
class StoredTranscript:
    chat_id: int
    message_id: int
    text: str
    confidence: int
    duration: float
    created_at: float

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.created_at)


class Storage:
    """Одно соединение + лок: запись в sqlite из нескольких корутин безопасна."""

    def __init__(self, path: Path = DB_PATH, ttl_seconds: float = TTL_SECONDS) -> None:
        self._path = path
        self._ttl = ttl_seconds
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._db is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        log.info("sqlite готова: %s (TTL %.0f ч)", self._path.resolve(), self._ttl / 3600)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Storage.connect() не вызван")
        return self._db

    async def save(
        self,
        *,
        chat_id: int,
        message_id: int,
        user_id: int | None,
        text: str,
        confidence: int,
        duration: float,
    ) -> None:
        async with self._lock:
            await self._conn().execute(
                """
                INSERT INTO transcripts
                    (chat_id, message_id, user_id, text, confidence, duration, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                    text       = excluded.text,
                    confidence = excluded.confidence,
                    duration   = excluded.duration,
                    created_at = excluded.created_at
                """,
                (chat_id, message_id, user_id, text, confidence, duration, time.time()),
            )
            await self._conn().commit()

    async def get(self, chat_id: int, message_id: int) -> StoredTranscript | None:
        """Возвращает запись, если она есть И не протухла (протухшую удаляет)."""
        async with self._lock:
            cursor = await self._conn().execute(
                "SELECT * FROM transcripts WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            if time.time() - row["created_at"] > self._ttl:
                await self._conn().execute(
                    "DELETE FROM transcripts WHERE chat_id = ? AND message_id = ?",
                    (chat_id, message_id),
                )
                await self._conn().commit()
                return None
        return StoredTranscript(
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            text=row["text"],
            confidence=row["confidence"],
            duration=row["duration"],
            created_at=row["created_at"],
        )

    async def purge_expired(self) -> int:
        async with self._lock:
            cutoff = time.time() - self._ttl
            cursor = await self._conn().execute(
                "DELETE FROM transcripts WHERE created_at < ?", (cutoff,)
            )
            removed = cursor.rowcount or 0
            await cursor.close()
            await self._conn().commit()
        return removed

    async def cleanup_loop(self, interval: float = CLEANUP_INTERVAL) -> None:
        """Фоновая чистка протухших записей (раз в час по умолчанию)."""
        while True:
            try:
                removed = await self.purge_expired()
                if removed:
                    log.info("чистка TTL: удалено %d транскриптов", removed)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ошибка в фоновой чистке транскриптов")
            await asyncio.sleep(interval)


storage = Storage()
