# QuantAI Restaurant Deployment Guide for Oracle Cloud

This guide explains how to deploy the QuantAI Restaurant system with Fonoster voice integration to Oracle Cloud Infrastructure (OCI).

## Prerequisites

1. An Oracle Cloud account (sign up at [cloud.oracle.com](https://cloud.oracle.com))
2. A Fonoster account (sign up at [fonoster.com](https://fonoster.com))
3. Your API keys for Groq, OpenAI, and Fonoster

## Step 1: Set Up Oracle Cloud Infrastructure (OCI)

### Create Compute Instance

1. Log in to your Oracle Cloud account
2. Navigate to **Compute** > **Instances** > **Create Instance**
3. Configure your instance:
   - **Name**: `quantai-restaurant`
   - **Image**: Oracle Linux 8
   - **Shape**: VM.Standard.E2.1 (1 OCPU, 8GB RAM) - Free tier eligible
   - **Network**: Create new VCN or use existing
   - **Subnet**: Public subnet
   - **Assign Public IP**: Yes
   - **SSH Keys**: Generate or upload your SSH key

### Configure Security Rules

1. Navigate to **Networking** > **Virtual Cloud Networks** > [Your VCN] > **Security Lists**
2. Click on the Default Security List
3. Add Ingress Rules for:
   - SSH (TCP port 22)
   - HTTP (TCP port 80)
   - HTTPS (TCP port 443)
   - FastAPI (TCP port 8006)
   - Fonoster Voice (TCP port 50061)

## Step 2: Connect to Your Instance

```bash
ssh opc@YOUR_INSTANCE_IP -i /path/to/your/private_key
```

## Step 3: Install Docker and Dependencies

```bash
# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker-engine

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group
sudo usermod -aG docker opc
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install other utilities
sudo yum install -y git unzip
```

## Step 4: Deploy the Application

### Option 1: Clone from GitHub (if repository is available)

```bash
# Clone the repository
git clone YOUR_REPOSITORY_URL
cd quantai-restaurant/backend
```

### Option 2: Upload Local Files

From your local machine:

```bash
# Create a zip archive
cd path/to/restaurant_v2
zip -r backend.zip backend

# Upload to Oracle Cloud
scp -i /path/to/your/private_key backend.zip opc@YOUR_INSTANCE_IP:~
```

On the Oracle Cloud instance:

```bash
# Extract the archive
unzip backend.zip
cd backend
```

### Configure Environment Variables

```bash
# Copy example config
cp fonoster_config.env.example .env

# Edit the .env file
nano .env
```

Add your credentials and update the `VOICE_APP_URL` to use your instance's public IP:

```
# LLM API keys
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key
ELEVENLABS_API_KEY=your-elevenlabs-api-key

# Fonoster credentials
FONOSTER_API_KEY=your-fonoster-api-key
FONOSTER_API_SECRET=your-fonoster-api-secret
FONOSTER_ACCESS_KEY_ID=your-fonoster-access-key-id
FONOSTER_FROM_NUMBER=your-fonoster-number
FONOSTER_VOICE_APP_REF=quantai-restaurant-app

# The public IP of your Oracle Cloud instance
VOICE_APP_URL=tcp://YOUR_ORACLE_CLOUD_IP:50061
```

### Build and Start Containers

```bash
# Run the deployment script
./deploy.sh
```

The script will:
1. Check for the .env file
2. Build and start the Docker containers
3. Update your Fonoster configuration with your public IP
4. Display the URLs for your services

## Step 5: Set Up Automatic Health Checks

```bash
# Make the health check script executable
chmod +x health_check.sh

# Add to crontab to run every 5 minutes
(crontab -l 2>/dev/null; echo "*/5 * * * * cd $(pwd) && ./health_check.sh") | crontab -
```

## Step 6: Set Up Nginx (Optional)

If you want to use a domain name and SSL:

```bash
# Install Nginx
sudo yum install -y nginx

# Create Nginx configuration
sudo nano /etc/nginx/conf.d/quantai.conf
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8006;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://localhost:8006;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

```bash
# Install Certbot for SSL
sudo yum install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Test renewal
sudo certbot renew --dry-run

# Start and enable Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

## Testing Your Deployment

1. Test FastAPI server:
   ```bash
   curl http://YOUR_PUBLIC_IP:8006/health
   ```

2. Test Fonoster connection:
   ```bash
   # Check if port is open
   nc -z -v YOUR_PUBLIC_IP 50061
   
   # Check Fonoster status
   curl http://YOUR_PUBLIC_IP:8006/fonoster/status
   ```

3. Make a test call to your Fonoster phone number

## Maintenance and Updates

### Viewing Logs

```bash
# View application logs
docker-compose logs -f

# View health check logs
cat logs/health_check.log
```

### Updating the Application

When you have new code to deploy:

```bash
# Pull the latest changes (if using git)
git pull

# Or upload and extract new files

# Then run the deployment script
./deploy.sh
```

### Monitoring Resources

```bash
# Check container status
docker-compose ps

# Check resource usage
docker stats

# Check disk space
df -h
```

## Troubleshooting

### Container Issues

```bash
# Restart containers
docker-compose restart

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Fonoster Connection Issues

```bash
# Check Fonoster CLI
fonoster login
fonoster apps:list

# Verify app configuration
fonoster apps:get YOUR_APP_REF

# Update voice URL
fonoster apps:update YOUR_APP_REF --voice-url=tcp://YOUR_PUBLIC_IP:50061
```

### Network Issues

```bash
# Check if ports are accessible
nc -z -v localhost 8006
nc -z -v localhost 50061

# Check Oracle Cloud security lists in the web console
```

## Backing Up Your Data

```bash
# Back up your data directory
sudo tar -czvf quantai-backup-$(date +%Y%m%d).tar.gz -C /path/to/backend/data .

# Copy to secure location
sudo cp quantai-backup-*.tar.gz /backup
```

## Need Help?

If you encounter any issues during deployment, please refer to:
- Docker documentation: [docs.docker.com](https://docs.docker.com)
- Oracle Cloud documentation: [docs.oracle.com](https://docs.oracle.com)
- Fonoster documentation: [docs.fonoster.com](https://docs.fonoster.com) 