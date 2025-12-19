import os
import time
import shutil
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import yaml
from faster_whisper import WhisperModel
from mutagen import MutagenError
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.oggvorbis import OggVorbis
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.ogg import OggFileType


def setup_logger(config: dict):
    log_level = getattr(logging, config["log_level"].upper())
    logger = logging.getLogger("whisper_processor")
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
    )

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
        return audio.info.length if hasattr(audio.info, 'length') else 0.0
    except Exception:
        return 0.0


def main():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config)

    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    temp_dir = Path(config.get("temp_dir", "./temp"))
    failed_dir = Path(config["move_failed_to"]) if config["move_failed_to"] else None

    for p in [input_dir, output_dir, temp_dir]:
        p.mkdir(parents=True, exist_ok=True)
    if failed_dir:
        failed_dir.mkdir(parents=True, exist_ok=True)

    extensions = tuple(ext.lower() for ext in config["supported_extensions"])

    logger.info("Загрузка модели faster-whisper...")
    try:
        model = WhisperModel(
            config["model_size"],
            device=config["device"],
            compute_type=config["compute_type"],
        )
        logger.info(f"Модель {config['model_size']} загружена на {config['device'].upper()} ({config['compute_type']})")
    except Exception as e:
        logger.critical(f"Не удалось загрузить модель: {e}")
        return

    logger.info("Faster-Whisper процессор запущен. Ожидаю файлы в ./input...")

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

    if config["vad_filter"]:
        transcribe_options["vad_filter"] = True
        transcribe_options["vad_parameters"] = config["vad_parameters"]

    while True:
        try:
            files = sorted(
                [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions],
                key=lambda x: x.stat().st_mtime,
            )

            if not files:
                time.sleep(config["poll_interval"])
                continue

            file_path = files[0]
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            duration_sec = get_audio_duration(file_path)
            if duration_sec <= 0:
                duration_sec = 0.0
            duration_str = format_duration(duration_sec)

            logger.info(f"Обнаружен файл: {file_path.name} | "
                        f"Размер: {file_size_mb:.1f} МБ | "
                        f"Длительность: {duration_str}")

            try:
                start_time = time.time()
                logger.info(f"Транскрибция -> {file_path.name}")

                segments, info = model.transcribe(str(file_path), **transcribe_options)

                if duration_sec <= 0 and info.duration:
                    duration_sec = info.duration
                    duration_str = format_duration(duration_sec)

                processing_time = time.time() - start_time

                text = " ".join(segment.text.strip() for segment in segments).strip()

                output_path = output_dir / f"{file_path.stem}.txt"
                output_path.write_text(text, encoding="utf-8")

                if duration_sec > 0:
                    rtf = duration_sec / processing_time if processing_time > 0 else float('inf')
                    speed_str = f"{rtf:.1f}x"
                else:
                    speed_str = "неизв."

                # ИСПРАВЛЕНО: правильное форматирование вероятности
                lang_prob_str = f"{info.language_probability:.2f}" if info.language_probability is not None else "0.00"

                logger.info(f"Успех -> {output_path.name} | "
                            f"Время обработки: {format_duration(processing_time)} | "
                            f"Скорость: {speed_str} | "
                            f"Язык: {info.language or 'авто'} (вероятность: {lang_prob_str})")

                if config["delete_input_after_process"]:
                    file_path.unlink()
                    logger.debug(f"Исходный файл удалён: {file_path.name}")

            except Exception as e:
                processing_time = time.time() - start_time if 'start_time' in locals() else 0
                logger.error(
                    f"Ошибка при обработке {file_path.name} (затрачено {format_duration(processing_time)}): {e}",
                    exc_info=True)
                if failed_dir:
                    dest = failed_dir / file_path.name
                    shutil.move(str(file_path), str(dest))
                    logger.warning(f"Файл перемещён в папку ошибок: {dest}")

        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки. Завершение...")
            break
        except Exception as e:
            logger.critical(f"Критическая ошибка в цикле: {e}", exc_info=True)
            time.sleep(10)

    logger.info("Работа завершена.")


if __name__ == "__main__":
    main()
