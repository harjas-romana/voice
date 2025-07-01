# Use Python 3.10 as base image
FROM python:3.10.18

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    curl \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs temp_audio

# Add a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose ports (FastAPI and Twilio webhook server)
EXPOSE 8006 5000

# Healthcheck (FastAPI health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8006/health || exit 1

# Entrypoint script for robust startup
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"] 