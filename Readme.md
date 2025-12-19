# WisperMVP

Минимальный сервис для офлайн-транскрибации аудио с помощью **faster-whisper**.

Принцип работы простой:
- кладёте аудиофайл в папку `input`
- сервис автоматически обрабатывает его
- результат сохраняется в `output`
- исходный аудиофайл удаляется

Интернет **не требуется** во время работы сервиса.

---

## Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone https://github.com/Dima51Al/WisperMVP

cd WisperMVP
```

Подготовка модели (делается один раз)
faster-whisper НЕ работает с .pt моделями.
Модель должна быть в формате CTranslate2.

Структура папки с моделью
```pgsql
models/
└── small/
    ├── model.bin
    ├── config.json
    └── vocabulary.json
```
### Вариант A: сконвертировать модель самостоятельно
Требуется интернет только на этом этапе.
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper ctranslate2 transformers
```

```bash

ct2-transformers-converter \
  --model openai/whisper-small \
  --output_dir models/small \
  --quantization int8
 ```

## Запуск сервиса
1. Запуск через Docker

```bash
docker compose up -d --build
```
2. Просмотр логов

```bash
docker logs -f wispermvp-whisper-1
```
## Использование
1) Поместите аудиофайл в папку input
2) Дождитесь обработки
3) Заберите текстовый файл из output

Поддерживаемые форматы задаются в config.yaml.

---

Примечания
сервис работает полностью офлайн

модель загружается из ./models

для CPU рекомендуется compute_type: int8



---