# Backend API Dockerfile — multi-stage pour réduire la taille finale

# Stage 1 : compilation des dépendances
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch (~220 MB) instead of the 6 GB CUDA default
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0,<2.5"
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --prefix=/install -r requirements.txt

# Stage 2 : image finale légère
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY . .

# Create cache directory with open permissions so both API (root) and docker exec work
RUN mkdir -p /app/.cache/huggingface && chmod 777 /app/.cache/huggingface

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
