FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./


RUN uv sync --frozen --no-install-project

COPY . .

RUN uv sync --frozen

EXPOSE 8000 8501

CMD ["uv", "run", "python", "main.py"]
