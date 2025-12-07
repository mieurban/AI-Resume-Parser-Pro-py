#!/bin/bash
# AWS EC2 Deployment Script for AI Resume Parser API
# Run this on a fresh Amazon Linux 2023 or Ubuntu EC2 instance

set -e

echo "=== AI Resume Parser API - AWS Deployment ==="

# Configuration
API_KEY="${API_KEY:-your-secure-api-key-here}"
PORT="${PORT:-8001}"

# Detect OS and install Docker
if [ -f /etc/amazon-linux-release ]; then
    echo "Installing Docker on Amazon Linux..."
    sudo yum update -y
    sudo yum install -y docker git
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ec2-user
elif [ -f /etc/lsb-release ]; then
    echo "Installing Docker on Ubuntu..."
    sudo apt update
    sudo apt install -y docker.io docker-compose git
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker ubuntu
fi

# Create app directory
APP_DIR="/opt/ai-resume-parser"
sudo mkdir -p $APP_DIR
sudo chown $(whoami):$(whoami) $APP_DIR
cd $APP_DIR

echo "=== Building Docker Image ==="
# If running from git clone, the files should be in place
# Otherwise, copy files manually to $APP_DIR

# Build the Docker image
sudo docker build -t ai-resume-parser .

echo "=== Stopping existing container (if any) ==="
sudo docker stop ai-resume-parser 2>/dev/null || true
sudo docker rm ai-resume-parser 2>/dev/null || true

echo "=== Starting AI Resume Parser Container ==="
sudo docker run -d \
    --name ai-resume-parser \
    --restart unless-stopped \
    -p ${PORT}:8001 \
    -e API_KEY="${API_KEY}" \
    -e LOG_LEVEL=info \
    -e HOST=0.0.0.0 \
    ai-resume-parser

echo "=== Waiting for container to be healthy ==="
sleep 10

# Check health
if curl -sf http://localhost:${PORT}/api/v1/health > /dev/null; then
    echo "✅ API is running and healthy!"
    echo "   Endpoint: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):${PORT}"
else
    echo "⚠️  Container started but health check failed. Check logs:"
    sudo docker logs ai-resume-parser
fi

echo ""
echo "=== Useful Commands ==="
echo "View logs:     sudo docker logs -f ai-resume-parser"
echo "Stop:          sudo docker stop ai-resume-parser"
echo "Start:         sudo docker start ai-resume-parser"
echo "Restart:       sudo docker restart ai-resume-parser"
echo "Shell access:  sudo docker exec -it ai-resume-parser bash"
echo ""
echo "=== Security Reminder ==="
echo "1. Configure your EC2 Security Group to allow inbound on port ${PORT}"
echo "2. Consider using HTTPS with a reverse proxy (nginx/caddy)"
echo "3. Restrict access to your Laravel app's IP if possible"
