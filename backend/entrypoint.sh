#!/bin/bash
set -e

# Start FastAPI server in the background
python server.py > logs/fastapi.log 2>&1 &
FASTAPI_PID=$!

# Start Twilio Flask server in the background
python twilio_voice_app.py > logs/twilio.log 2>&1 &
TWILIO_PID=$!

# Wait for both processes
wait $FASTAPI_PID
wait $TWILIO_PID 