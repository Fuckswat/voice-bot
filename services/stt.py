"""Обёртка над faster-whisper.

Модель грузится один раз (ленивo или через warmup на старте бота).
Инференс синхронный и держит GIL, поэтому вызывается через asyncio.to_thread —
event loop не блокируется и бот продолжает отвечать другим юзерам.
Сам инференс сериализуется семафором (по умолчанию 1 одновременный прогон):
одна WhisperModel не рассчитана на параллельные вызовы, а очередь дешевле,
чем вторая копия medium в памяти.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
CONCURRENCY = max(1, int(os.getenv("STT_CONCURRENCY", "1")))

# Границы нормализации «сырого» качества сегмента в проценты.
# Ниже RAW_FLOOR — распознавание мусорное, выше RAW_CEIL — практически идеальное.
RAW_FLOOR = 0.15
RAW_CEIL = 0.88

_model: WhisperModel | None = None
_model_lock = asyncio.Lock()
_infer_semaphore = asyncio.Semaphore(CONCURRENCY)
_active_device: tuple[str, str] = ("?", "?")  # что реально используется сейчас
_rebuild_lock = threading.Lock()

# признаки того, что упала именно CUDA-часть, а не распознавание
_CUDA_ERROR_MARKERS = ("cublas", "cudnn", "cuda", "gpu", "nvidia")


@dataclass(slots=True)
class Transcription:
    text: str
    confidence: int  # 0..100
    duration: float  # секунды аудио по данным whisper
    segments: int

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _has_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _pick_device() -> tuple[str, str]:
    """(device, compute_type). WHISPER_DEVICE=cpu|cuda принудительно перекрывает детект."""
    forced = os.getenv("WHISPER_DEVICE", "").strip().lower()
    if forced == "cuda":
        return "cuda", os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    if forced == "cpu":
        return "cpu", os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    if _has_nvidia_gpu():
        return "cuda", os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    return "cpu", os.getenv("WHISPER_COMPUTE_TYPE", "int8")


def _register_nvidia_dll_dirs() -> None:
    """Windows: подсунуть CTranslate2 DLL из колёс nvidia-*-cu12.

    Без этого cublas64_12.dll/cudnn64_9.dll не находятся, даже когда пакеты стоят
    в venv. Одного os.add_dll_directory() мало: CTranslate2 зовёт обычный
    LoadLibrary, который смотрит в PATH, а не в user-directories загрузчика —
    поэтому правим и то, и другое.
    """
    if os.name != "nt":
        return
    try:
        import nvidia
    except ImportError:
        return

    found: list[str] = []
    for root in getattr(nvidia, "__path__", []):
        for dll_dir in Path(root).glob("*/bin"):
            if not any(dll_dir.glob("*.dll")):
                continue
            path = str(dll_dir)
            found.append(path)
            try:
                os.add_dll_directory(path)
            except OSError:
                pass

    if found:
        os.environ["PATH"] = os.pathsep.join(found) + os.pathsep + os.environ.get("PATH", "")
        log.debug("добавил в PATH CUDA-библиотеки: %s", ", ".join(found))


def _build(device: str, compute_type: str) -> WhisperModel:
    global _active_device
    log.info("загружаю whisper %s (device=%s, compute_type=%s)", MODEL_SIZE, device, compute_type)
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
    _active_device = (device, compute_type)
    return model


def _load_model() -> WhisperModel:
    _register_nvidia_dll_dirs()
    device, compute_type = _pick_device()
    try:
        return _build(device, compute_type)
    except Exception:
        if device == "cpu":
            raise
        # Типичный случай на Windows: nvidia-smi есть, а cuDNN/cuBLAS для CTranslate2 не стоят.
        log.warning("cuda недоступна для CTranslate2, откатываюсь на cpu/int8", exc_info=True)
        return _build("cpu", "int8")


def _looks_like_cuda_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CUDA_ERROR_MARKERS)


async def get_model() -> WhisperModel:
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                _model = await asyncio.to_thread(_load_model)
    return _model


async def warmup() -> None:
    """Прогреть модель на старте, чтобы первый юзер не ждал загрузку весов."""
    await get_model()


def _segment_score(avg_logprob: float, no_speech_prob: float) -> float:
    """Сырое качество сегмента в 0..1.

    exp(avg_logprob) — средняя вероятность токена (0.85 для чистой речи,
    0.35-0.5 для шумной), домножается на (1 - no_speech_prob), чтобы
    «распознанная тишина» роняла оценку.
    """
    prob = math.exp(max(-10.0, min(0.0, avg_logprob)))
    speech = 1.0 - max(0.0, min(1.0, no_speech_prob))
    return prob * speech


def _to_percent(raw: float) -> int:
    scaled = (raw - RAW_FLOOR) / (RAW_CEIL - RAW_FLOOR)
    return int(round(max(0.0, min(1.0, scaled)) * 100))


def _run_model(model: WhisperModel, path: str) -> Transcription:
    segments, info = model.transcribe(
        path,
        language=LANGUAGE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=BEAM_SIZE,
        condition_on_previous_text=False,
    )

    parts: list[str] = []
    weighted_sum = 0.0
    weight_total = 0.0
    count = 0

    for seg in segments:  # генератор — распознавание идёт здесь
        text = (seg.text or "").strip()
        if not text:
            continue
        count += 1
        parts.append(text)
        weight = max(0.1, float(seg.end) - float(seg.start))
        weighted_sum += _segment_score(float(seg.avg_logprob), float(seg.no_speech_prob)) * weight
        weight_total += weight

    full_text = " ".join(parts).strip()
    if not full_text or weight_total == 0:
        return Transcription(text="", confidence=0, duration=float(info.duration), segments=0)

    return Transcription(
        text=full_text,
        confidence=_to_percent(weighted_sum / weight_total),
        duration=float(info.duration),
        segments=count,
    )


def _transcribe_sync(path: str) -> Transcription:
    global _model

    model = _model
    if model is None:  # на случай прямого вызова в обход get_model()
        model = _model = _load_model()

    try:
        return _run_model(model, path)
    except RuntimeError as exc:
        # CTranslate2 грузит cuBLAS/cuDNN лениво: модель создалась на cuda,
        # а падает только здесь, на первом инференсе. Переезжаем на CPU насовсем.
        if _active_device[0] != "cuda" or not _looks_like_cuda_failure(exc):
            raise
        log.warning("инференс на cuda упал (%s), перехожу на cpu/int8 до перезапуска", exc)
        with _rebuild_lock:
            if _active_device[0] == "cuda":
                _model = _build("cpu", "int8")
            model = _model
        return _run_model(model, path)


async def transcribe(path: str) -> Transcription:
    """Распознать файл. Не блокирует event loop."""
    await get_model()
    async with _infer_semaphore:
        return await asyncio.to_thread(_transcribe_sync, path)
