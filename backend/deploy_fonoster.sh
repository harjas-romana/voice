#!/bin/bash
# QuantAI Restaurant Fonoster Deployment Script
# =============================================
# This script helps deploy the QuantAI Restaurant system with Fonoster
# voice calling integration.

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}QuantAI Restaurant Fonoster Deployment Script${NC}"
echo "=================================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from example...${NC}"
    if [ -f fonoster_config.env.example ]; then
        cp fonoster_config.env.example .env
        echo -e "${GREEN}Created .env file. Please edit it with your credentials.${NC}"
    else
        echo -e "${RED}No fonoster_config.env.example file found. Please create .env manually.${NC}"
        exit 1
    fi
fi

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Node.js is not installed. Installing...${NC}"
    # Install Node.js using NVM (Node Version Manager)
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    nvm install 18
    nvm use 18
fi

# Install Fonoster CLI
echo -e "${GREEN}Installing Fonoster CLI...${NC}"
npm install -g @fonoster/ctl

# Check for Fonoster credentials
echo -e "${GREEN}Checking Fonoster credentials...${NC}"
if grep -q "FONOSTER_API_KEY" .env && grep -q "FONOSTER_API_SECRET" .env && grep -q "FONOSTER_ACCESS_KEY_ID" .env; then
    echo -e "${GREEN}Fonoster credentials found in .env file.${NC}"
else
    echo -e "${YELLOW}Fonoster credentials not found in .env file.${NC}"
    echo "Please edit the .env file and add your Fonoster credentials."
    echo "Then run this script again."
    exit 1
fi

# Set up Fonoster resources
echo -e "${GREEN}Setting up Fonoster resources...${NC}"
python -c "import asyncio; from fonoster_config import setup_fonoster; asyncio.run(setup_fonoster())"

# Set up systemd service for the voice server (if running on Linux)
if [ "$(uname)" == "Linux" ]; then
    echo -e "${GREEN}Setting up systemd service for the voice server...${NC}"
    
    # Create service file
    SERVICE_FILE="/tmp/quantai-voice.service"
    cat > $SERVICE_FILE << EOL
[Unit]
Description=QuantAI Restaurant Voice Server
After=network.target

[Service]
User=$(whoami)
WorkingDirectory=$(pwd)
ExecStart=$(which python) fonoster_voice_app.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=$(pwd)/.env

[Install]
WantedBy=multi-user.target
EOL

    # Ask for sudo permission to copy service file
    echo "This will install the systemd service for the voice server."
    echo "You will need sudo privileges to continue."
    sudo cp $SERVICE_FILE /etc/systemd/system/quantai-voice.service
    sudo systemctl daemon-reload
    sudo systemctl enable quantai-voice.service
    
    echo -e "${GREEN}Systemd service created. To start the service, run:${NC}"
    echo "sudo systemctl start quantai-voice.service"
else
    echo -e "${YELLOW}Not running on Linux. Skipping systemd service setup.${NC}"
    echo "To start the voice server, run:"
    echo "python fonoster_voice_app.py"
fi

# Start the FastAPI server
echo -e "${GREEN}Starting the FastAPI server...${NC}"
echo "To start the server, run:"
echo "python server.py"

echo ""
echo -e "${GREEN}Deployment complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your credentials if you haven't already."
echo "2. Start the FastAPI server: python server.py"
echo "3. Start the voice server: python fonoster_voice_app.py"
echo "4. Make a test call to your Fonoster phone number."
echo ""
echo "For more information, see the README.md file." 