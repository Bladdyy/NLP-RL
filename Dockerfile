FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv by copying it directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN uv venv /app/env
ENV PATH="/app/env/bin:$PATH"

COPY pyproject.toml uv.lock* ./

RUN uv pip install --no-cache-dir -r pyproject.toml

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/env /app/env
COPY . /app

ENV PATH="/app/env/bin:$PATH"

RUN chmod -R 755 /app