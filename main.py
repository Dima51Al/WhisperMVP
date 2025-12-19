import os
import whisper
import torch
from pathlib import Path
import yaml
import time

with open("config.yaml") as f:
    config = yaml.safe_load(f)

input_dir = Path(config["input_dir"])
output_dir = Path(config["output_dir"])
output_dir.mkdir(exist_ok=True)

model = whisper.load_model(config["model_path"], device=config["device"])

print("Whisper готов. Ожидаю файлы в input/")

while True:
    files = sorted([f for f in input_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}])

    if not files:
        time.sleep(2)
        continue

    file_path = files[0]
    print(f"Обрабатываю: {file_path.name}")

    try:
        result = model.transcribe(
            str(file_path),
            language=config["language"],
            fp16=config["device"] == "cuda"
        )
        text = result["text"].strip()

        output_file = output_dir / (file_path.stem + ".txt")
        output_file.write_text(text, encoding="utf-8")

        file_path.unlink()  # удаляем исходный файл
        print(f"Готово → {output_file.name}")

    except Exception as e:
        print(f"Ошибка при обработке {file_path.name}: {e}")
        time.sleep(5)