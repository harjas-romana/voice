#!/bin/bash
# Deployment script for QuantAI Restaurant

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}QuantAI Restaurant Deployment Script${NC}"
echo "====================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from example...${NC}"
    if [ -f fonoster_config.env.example ]; then
        cp fonoster_config.env.example .env
        echo -e "${GREEN}Created .env file. Please edit it with your credentials.${NC}"
        echo -e "${RED}IMPORTANT: Edit the .env file before continuing!${NC}"
        exit 1
    else
        echo -e "${RED}No fonoster_config.env.example file found. Please create .env manually.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Stopping existing containers...${NC}"
docker-compose down

echo -e "${GREEN}Building containers...${NC}"
docker-compose build

echo -e "${GREEN}Starting containers...${NC}"
docker-compose up -d

echo -e "${GREEN}Updating Fonoster configuration...${NC}"
# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me)
echo "Public IP: $PUBLIC_IP"

# Check if Fonoster CLI is installed
if ! command -v fonoster &> /dev/null; then
    echo -e "${YELLOW}Fonoster CLI not installed. Installing...${NC}"
    npm install -g @fonoster/ctl
fi

# Login to Fonoster using credentials from .env
source .env
echo -e "${YELLOW}Logging into Fonoster...${NC}"
echo -e "${YELLOW}You may need to manually enter your credentials${NC}"
fonoster login

# List Fonoster apps
echo -e "${GREEN}Listing Fonoster apps...${NC}"
fonoster apps:list

# Prompt for app reference
echo -e "${YELLOW}Enter your Fonoster app reference from the list above:${NC}"
read APP_REF

# Update the app with the public URL
echo -e "${GREEN}Updating Fonoster app with public URL...${NC}"
fonoster apps:update $APP_REF --voice-url="tcp://$PUBLIC_IP:50061"

echo -e "${GREEN}Checking container status...${NC}"
docker-compose ps

echo -e "${GREEN}Deployment complete!${NC}"
echo "====================================="
echo "Your QuantAI Restaurant system is now deployed!"
echo "FastAPI server: http://$PUBLIC_IP:8006"
echo "Voice server: tcp://$PUBLIC_IP:50061" 