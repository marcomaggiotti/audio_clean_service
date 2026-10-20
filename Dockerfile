FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY main.py .
# Uploads, cleaned audio and the job list live under /data (a Render persistent disk in render.yaml;
# mount a volume there elsewhere). 10000 is Render's default port; PORT, when set, overrides it.
ENV DATA_DIR=/data \
    PORT=10000
EXPOSE 10000
CMD ["python", "main.py"]
