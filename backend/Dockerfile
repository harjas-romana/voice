# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Fonoster CLI
RUN npm install -g @fonoster/ctl

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs temp_audio

# Expose ports (FastAPI and Fonoster Voice Server)
EXPOSE 8006 50061

# Create entrypoint script
RUN echo '#!/bin/bash\n\
python server.py & \n\
python fonoster_voice_app.py\n\
wait' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 