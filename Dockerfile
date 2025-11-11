# Candle vLLM Studio container image
ARG BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies, Node.js (LTS) and tooling required to build candle-vllm
RUN apt-get update && apt-get install -y \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        curl \
        wget \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        libcurl4-openssl-dev \
        unzip \
        ca-certificates \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js LTS (required for the Vue frontend build)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for candle-vllm source builds
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y --profile minimal \
    && /opt/cargo/bin/rustup component add rustfmt clippy
ENV PATH="/opt/cargo/bin:${PATH}"

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Frontend build arguments
ARG BUILD_FRONTEND=true

# Prime npm cache layer (root package.json manages frontend build)
WORKDIR /app
COPY package.json package-lock.json* ./
RUN if [ "$BUILD_FRONTEND" = "true" ]; then npm install; else echo "Skipping npm install"; fi

# Copy backend code and frontend sources (built inside container)
COPY backend/ ./backend/
COPY frontend/src ./frontend/src
COPY frontend/public ./frontend/public
COPY frontend/index.html ./frontend/index.html
COPY frontend/vite.config.js ./frontend/vite.config.js
COPY README.md .

# Build the frontend bundle (optional for faster backend-only iteration)
RUN if [ "$BUILD_FRONTEND" = "true" ]; then npm run build; else echo "Skipping frontend build"; fi

# Ensure python command exists for base images where it's absent
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create non-root user and data directory layout
RUN useradd -ms /bin/bash appuser \
    && mkdir -p /app/data/{models,configs,logs,candle-builds} \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /opt/cargo /opt/rustup

# Runtime configuration
USER appuser
EXPOSE 8080
VOLUME ["/app/data"]

# Start FastAPI application
CMD ["python", "backend/main.py"]

