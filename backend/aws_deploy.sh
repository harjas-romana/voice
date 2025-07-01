#!/bin/bash
# AWS Deployment script for QuantAI Restaurant with Twilio

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}QuantAI Restaurant AWS Deployment Script${NC}"
echo "============================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from example...${NC}"
    if [ -f twilio_env.example ]; then
        cp twilio_env.example .env
        echo -e "${GREEN}Created .env file. Please edit it with your credentials.${NC}"
        echo -e "${RED}IMPORTANT: Edit the .env file before continuing!${NC}"
        exit 1
    else
        echo -e "${RED}No twilio_env.example file found. Please create .env manually.${NC}"
        exit 1
    fi
fi

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Get public IP address
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com/)
if [ -z "$PUBLIC_IP" ]; then
    echo -e "${YELLOW}Could not determine public IP automatically.${NC}"
    read -p "Please enter the public IP of this server: " PUBLIC_IP
fi

echo -e "${GREEN}Detected public IP: ${PUBLIC_IP}${NC}"

# Update .env file with the public IP if needed
if grep -q "TWILIO_WEBHOOK_BASE_URL" .env; then
    # Update existing variable
    sed -i.bak "s|TWILIO_WEBHOOK_BASE_URL=.*|TWILIO_WEBHOOK_BASE_URL=http://${PUBLIC_IP}:5000|g" .env
else
    # Add the variable
    echo -e "\n# Server public IP address" >> .env
    echo "TWILIO_WEBHOOK_BASE_URL=http://${PUBLIC_IP}:5000" >> .env
fi

echo -e "${GREEN}Updated .env with public IP address${NC}"

# Building and starting containers
echo -e "${GREEN}Stopping existing containers (if any)...${NC}"
docker compose down || docker-compose down

echo -e "${GREEN}Building containers...${NC}"
docker compose build || docker-compose build

echo -e "${GREEN}Starting containers...${NC}"
docker compose up -d || docker-compose up -d

echo -e "${GREEN}Containers started successfully!${NC}"

# Wait for services to start
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Setup Twilio webhooks
echo -e "${GREEN}Setting up Twilio webhooks...${NC}"
curl -s -X POST "http://localhost:8006/twilio/setup" | grep -q "status.*ok" && \
    echo -e "${GREEN}Twilio webhook setup successful!${NC}" || \
    echo -e "${RED}Twilio webhook setup may have failed. Please check logs.${NC}"

# Check status
echo -e "${GREEN}Checking Twilio status...${NC}"
curl -s "http://localhost:8006/twilio/status"
echo ""

echo -e "${GREEN}Deployment completed!${NC}"
echo "============================================"
echo -e "${YELLOW}Your server is now running at:${NC}"
echo -e "FastAPI: http://${PUBLIC_IP}:8006"
echo -e "Twilio Webhook Server: http://${PUBLIC_IP}:5000"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Ensure your Twilio phone number is configured correctly"
echo "2. Test your deployment by calling your Twilio number"
echo "3. Set up the health check script for monitoring:"
echo "   chmod +x health_check.sh && ./health_check.sh"
echo ""
echo -e "${GREEN}For support, check the logs in the logs/ directory${NC}" 