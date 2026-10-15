FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Agregamos flask
RUN pip install --no-cache-dir opencv-python-headless ultralytics numpy flask

CMD ["python3", "src/main_trucks.py"]
