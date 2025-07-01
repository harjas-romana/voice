# Use Python 3.10 as base image
FROM python:3.10.18

# Set working directory
WORKDIR /app

# Install system dependencies and latest Rust (for sudachipy/TTS build)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && rustc --version

# Ensure Rust is in PATH for all future RUN commands
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip, setuptools, and wheel, and preinstall sudachipy for TTS build reliability
RUN pip install --upgrade pip setuptools wheel
RUN pip install sudachipy==0.6.7

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs temp_audio

# Copy entrypoint script and set permissions as root
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Add a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose ports (FastAPI and Twilio webhook server)
EXPOSE 8006 5000

# Healthcheck (FastAPI health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8006/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"] 