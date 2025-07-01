#!/bin/bash
# Health check script for QuantAI Restaurant

LOG_FILE="logs/health_check.log"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "[$TIMESTAMP] Running health check..." >> $LOG_FILE

# Check FastAPI server
if curl -s "http://localhost:8006/health" | grep -q "healthy"; then
  echo "[$TIMESTAMP] FastAPI server is running" >> $LOG_FILE
else
  echo "[$TIMESTAMP] FastAPI server is down, restarting..." >> $LOG_FILE
  docker-compose restart quantai-restaurant
  echo "[$TIMESTAMP] Restart command sent" >> $LOG_FILE
fi

# Check Fonoster voice server
if nc -z localhost 50061 2>/dev/null; then
  echo "[$TIMESTAMP] Voice server is running" >> $LOG_FILE
else
  echo "[$TIMESTAMP] Voice server is down, restarting..." >> $LOG_FILE
  docker-compose restart quantai-restaurant
  echo "[$TIMESTAMP] Restart command sent" >> $LOG_FILE
fi

# Check system resources
MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" quantai-restaurant)
CPU_USAGE=$(docker stats --no-stream --format "{{.CPUPerc}}" quantai-restaurant)
DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}')

echo "[$TIMESTAMP] Memory usage: $MEMORY_USAGE" >> $LOG_FILE
echo "[$TIMESTAMP] CPU usage: $CPU_USAGE" >> $LOG_FILE
echo "[$TIMESTAMP] Disk usage: $DISK_USAGE" >> $LOG_FILE

# Alert if resources are critical
if [[ ${MEMORY_USAGE%\%} -gt 90 ]]; then
  echo "[$TIMESTAMP] WARNING: Memory usage is critical!" >> $LOG_FILE
fi

if [[ ${CPU_USAGE%\%} -gt 90 ]]; then
  echo "[$TIMESTAMP] WARNING: CPU usage is critical!" >> $LOG_FILE
fi

if [[ ${DISK_USAGE%\%} -gt 90 ]]; then
  echo "[$TIMESTAMP] WARNING: Disk usage is critical!" >> $LOG_FILE
fi

echo "[$TIMESTAMP] Health check complete" >> $LOG_FILE
echo "" >> $LOG_FILE 