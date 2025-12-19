FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Только ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY config.yaml .
COPY main.py .

VOLUME ["/app/models", "/app/input", "/app/output"]

CMD ["python", "main.py"]