# ============================================================
# Stage 1: Builder — install all dependencies with compilers
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies (removed in final stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into --user prefix
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ============================================================
# Stage 2: Final — slim runtime image
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Install ONLY runtime libraries, redis-server, and redis-tools
# libgl1 is the #1 missing dependency for OpenCV in Docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    redis-server \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput 2>/dev/null || true

# Expose port 8000 for standard environments
EXPOSE 8000

# Make start script executable
RUN chmod +x start.sh

# Run start script to boot Redis, Celery, and Django
CMD ["./start.sh"]
