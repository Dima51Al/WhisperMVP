import os
import time
import shutil
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import yaml

from faster_whisper import WhisperModel

from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.oggvorbis import OggVorbis
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.ogg import OggFileType


# =========================
# ЛОГГЕР
# =========================
def setup_logger(config: dict):
    log_level = getattr(logging, config["log_level"].upper())
    logger = logging.getLogger("whisper_processor")
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
    )

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        log_file = config.get("log_file")
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                log_file,
                maxBytes=config.get("log_max_bytes", 10 * 1024 * 1024),
                backupCount=config.get("log_backup_count", 5),
                encoding="utf-8",
            )
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# =========================
# ВСПОМОГАТЕЛЬНОЕ
# =========================
def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "неизвестно"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours} ч {minutes:02d} мин {secs:02d} сек"
    elif minutes > 0:
        return f"{minutes} мин {secs:02d} сек"
    else:
        return f"{secs} сек"


def get_audio_duration(file_path: Path) -> float:
    try:
        ext = file_path.suffix.lower()
        if ext == ".mp3":
            audio = MP3(file_path)
        elif ext == ".wav":
            audio = WAVE(file_path)
        elif ext in [".ogg", ".oga"]:
            audio = OggVorbis(file_path) or OggFileType(file_path)
        elif ext == ".flac":
            audio = FLAC(file_path)
        elif ext in [".m4a", ".mp4", ".mpeg", ".webm"]:
            audio = MP4(file_path)
        else:
            return 0.0
        return audio.info.length if hasattr(audio.info, "length") else 0.0
    except Exception:
        return 0.0


# =========================
# MAIN
# =========================
def main():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config)

    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    temp_dir = Path(config.get("temp_dir", "./temp"))
    failed_dir = Path(config["move_failed_to"]) if config.get("move_failed_to") else None

    for p in [input_dir, output_dir, temp_dir]:
        p.mkdir(parents=True, exist_ok=True)
    if failed_dir:
        failed_dir.mkdir(parents=True, exist_ok=True)

    extensions = tuple(ext.lower() for ext in config["supported_extensions"])

    # ===== ЗАГРУЗКА МОДЕЛИ =====
    logger.info("Загрузка модели faster-whisper...")
    try:
        model = WhisperModel(
            config["model_size"],
            device=config["device"],
            compute_type=config["compute_type"],
        )
        logger.info(
            f"Модель {config['model_size']} загружена на "
            f"{config['device'].upper()} ({config['compute_type']})"
        )
    except Exception as e:
        logger.critical(f"Не удалось загрузить модель: {e}")
        return

    # ===== ПАРАМЕТРЫ WHISPER (БЕЗ VAD!) =====
    transcribe_options = {
        "language": config["language"],
        "task": "translate" if config["translate"] else "transcribe",
        "beam_size": config["beam_size"],
        "temperature": tuple(config["temperature"]),
        "compression_ratio_threshold": config["compression_ratio_threshold"],
        "log_prob_threshold": config["logprob_threshold"],
        "no_speech_threshold": config["no_speech_threshold"],
        "condition_on_previous_text": config["condition_on_previous_text"],
        "initial_prompt": config["initial_prompt"],
        "word_timestamps": config["word_timestamps"],
    }

    # ===== ПАРАМЕТРЫ VAD (ОТДЕЛЬНО!) =====
    vad_filter = False
    vad_parameters = None

    if config.get("vad_filter"):
        vad_filter = True
        vad_cfg = config.get("vad_parameters", {})

        vad_parameters = {
            "min_speech_duration_ms": vad_cfg.get("min_speech_duration_ms", 250),
            "max_speech_duration_s": vad_cfg.get("max_speech_duration_s", 30),
            "min_silence_duration_ms": vad_cfg.get("min_silence_duration_ms", 2000),
            "speech_pad_ms": vad_cfg.get("speech_pad_ms", 400),
        }

        logger.info(f"VAD включён: {vad_parameters}")
    else:
        logger.info("VAD отключён")

    logger.info("Faster-Whisper процессор запущен. Ожидание файлов...")

    # ===== ОСНОВНОЙ ЦИКЛ =====
    while True:
        try:
            files = sorted(
                [
                    f
                    for f in input_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in extensions
                ],
                key=lambda x: x.stat().st_mtime,
            )

            if not files:
                time.sleep(config["poll_interval"])
                continue

            file_path = files[0]
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            duration_sec = get_audio_duration(file_path)
            duration_str = format_duration(duration_sec)

            logger.info(
                f"Обнаружен файл: {file_path.name} | "
                f"Размер: {file_size_mb:.1f} МБ | "
                f"Длительность: {duration_str}"
            )

            start_time = time.time()

            try:
                segments, info = model.transcribe(
                    str(file_path),
                    vad_filter=vad_filter,
                    vad_parameters=vad_parameters,
                    **transcribe_options,
                )

                # генератор → список
                segments = list(segments)

                processing_time = time.time() - start_time

                # ===== СПЛОШНОЙ ТЕКСТ =====
                full_text = " ".join(
                    s.text.strip() for s in segments if s.text.strip()
                )

                output_text_path = output_dir / f"{file_path.stem}.txt"
                output_text_path.write_text(full_text, encoding="utf-8")

                # ===== СЕГМЕНТЫ С ТАЙМКОДАМИ =====
                segments_path = output_dir / f"{file_path.stem}_segments.txt"
                with segments_path.open("w", encoding="utf-8") as f:
                    for s in segments:
                        text = s.text.strip()
                        if not text:
                            continue
                        f.write(f"[{s.start:.2f} - {s.end:.2f}] {text}\n")

                rtf = (
                    duration_sec / processing_time
                    if duration_sec > 0 and processing_time > 0
                    else 0
                )

                logger.info(
                    f"Успех -> {file_path.name} | "
                    f"Время: {format_duration(processing_time)} | "
                    f"Скорость: {rtf:.1f}x | "
                    f"Файлы: {output_text_path.name}, {segments_path.name}"
                )

                if config["delete_input_after_process"]:
                    file_path.unlink()

            except Exception as e:
                logger.error(
                    f"Ошибка обработки {file_path.name}: {e}", exc_info=True
                )
                if failed_dir:
                    shutil.move(
                        str(file_path),
                        str(failed_dir / file_path.name),
                    )

        except KeyboardInterrupt:
            logger.info("Остановка по Ctrl+C")
            break
        except Exception as e:
            logger.critical(f"Критическая ошибка: {e}", exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
