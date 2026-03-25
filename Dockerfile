# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency manifest first (maximises layer caching)
COPY pyproject.toml .

# Install only production dependencies into /app/.venv
RUN uv sync --no-dev –no-install-project

# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Bring the pre-built virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source and pre-trained model artifacts
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Prepend the venv to PATH so uvicorn resolves correctly
ENV PATH=/app/.venv/bin:$PATH

EXPOSE 8000

CMD ["uvicorn", "movie_reco_api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
