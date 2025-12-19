import os
import time
import shutil
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import yaml
import whisper
from whisper.utils import get_writer

# ====================== НАСТРОЙКА ЛОГГЕРА ======================
def setup_logger(config: dict):
    log_level = getattr(logging, config["log_level"].upper())
    logger = logging.getLogger("whisper_processor")
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
    )

    # Консоль
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Файл (если указан)
    log_file = config.get("log_file")
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_file,
            maxBytes=config.get("log_max_bytes", 10 * 1024 * 1024),
            backupCount=config.get("log_backup_count", 5),
        )
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ====================== ОСНОВНОЙ КОД ======================
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

    extensions = tuple(config["supported_extensions"])

    logger.info("Загрузка модели Whisper...")
    try:
        model = whisper.load_model(config["model_path"], device=config["device"])
        logger.info(f"Модель загружена на {config['device'].upper()}")
    except Exception as e:
        logger.critical(f"Не удалось загрузить модель: {e}")
        return

    logger.info("Whisper-процессор запущен. Ожидаю файлы...")

    transcribe_options = {
        "language": config["language"],
        "task": "translate" if config["translate"] else "transcribe",
        "beam_size": config["beam_size"],
        "best_of": config["best_of"],
        "patience": config["patience"],
        "temperature": tuple(config["temperature"]),
        "compression_ratio_threshold": config["compression_ratio_threshold"],
        "logprob_threshold": config["logprob_threshold"],
        "no_speech_threshold": config["no_speech_threshold"],
        "condition_on_previous_text": config["condition_on_previous_text"],
        "initial_prompt": config["initial_prompt"],
        "word_timestamps": config["word_timestamps"],
        "fp16": config["device"] == "cuda",
        "batch_size": config["batch_size"],
    }

    if config["vad_filter"]:
        transcribe_options["vad_filter"] = True
        transcribe_options["vad_parameters"] = config["vad_parameters"]

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
            logger.info(f"Обнаружен файл: {file_path.name}")

            try:
                logger.info(f"Транскрибция → {file_path.name}")
                result = model.transcribe(str(file_path), **transcribe_options)

                # Сохранение текста
                txt_writer = get_writer("txt", output_dir)
                txt_writer(result, str(file_path))

                # Если нужны SRT/VTT/JSON — раскомментируй
                # srt_writer = get_writer("srt", output_dir)
                # srt_writer(result, str(file_path))

                logger.info(f"Успех → {file_path.stem}.txt")

                # Удаление исходника
                if config["delete_input_after_process"]:
                    file_path.unlink()
                    logger.debug(f"Исходный файл удалён: {file_path.name}")

            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path.name}: {e}", exc_info=True)

                if failed_dir:
                    dest = failed_dir / file_path.name
                    shutil.move(str(file_path), str(dest))
                    logger.warning(f"Файл перемещён в папку ошибок: {dest}")
                # Иначе оставляем в input для повторной попытки

        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки. Завершение...")
            break
        except Exception as e:
            logger.critical(f"Критическая ошибка в основном цикле: {e}", exc_info=True)
            time.sleep(10)

    logger.info("Работа завершена.")


if __name__ == "__main__":
    main()