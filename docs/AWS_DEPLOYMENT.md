# AWS Deployment Guide for AI Resume Parser API

## Prerequisites
- AWS Account
- AWS CLI configured (`aws configure`)
- EC2 key pair for SSH access

---

## Option 1: EC2 Deployment (Recommended for Start)

### Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Settings:
   - **Name**: `ai-resume-parser`
   - **AMI**: Amazon Linux 2023 or Ubuntu 22.04
   - **Instance type**: `t3.small` (2 vCPU, 2GB RAM) - minimum for spaCy
   - **Key pair**: Select or create one
   - **Security Group**: Allow SSH (22) and Custom TCP (8001)
   - **Storage**: 20GB gp3

### Step 2: Connect and Deploy

```bash
# SSH into your instance
ssh -i your-key.pem ec2-user@<public-ip>

# Clone your repository
git clone https://github.com/yourusername/applymate.git
cd applymate/ai-resume-parser-api

# Set your API key
export API_KEY="your-secure-api-key"

# Run deployment script
chmod +x aws-deploy.sh
./aws-deploy.sh
```

### Step 3: Configure Security Group

1. EC2 → Security Groups → Your instance's security group
2. Add Inbound Rule:
   - **Type**: Custom TCP
   - **Port**: 8001
   - **Source**: Your Laravel app's IP or 0.0.0.0/0 (less secure)

### Step 4: Update Laravel Configuration

In your Laravel `.env`:
```
AI_PARSER_PRO_URL=http://<ec2-public-ip>:8001
AI_PARSER_PRO_API_KEY=your-secure-api-key
```

---

## Option 2: Using Docker Compose on EC2

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  ai-resume-parser:
    build: .
    container_name: ai-resume-parser
    restart: unless-stopped
    ports:
      - "8001:8001"
    environment:
      - API_KEY=${API_KEY}
      - LOG_LEVEL=info
      - HOST=0.0.0.0
      - PORT=8001
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

Run with:
```bash
API_KEY=your-key docker-compose -f docker-compose.prod.yml up -d
```

---

## Option 3: AWS ECS with Fargate (Production/Scalable)

### Step 1: Create ECR Repository

```bash
aws ecr create-repository \
    --repository-name ai-resume-parser \
    --region us-east-1
```

### Step 2: Build & Push Image

```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t ai-resume-parser .

# Tag for ECR
docker tag ai-resume-parser:latest \
    <account-id>.dkr.ecr.us-east-1.amazonaws.com/ai-resume-parser:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ai-resume-parser:latest
```

### Step 3: Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name applymate-cluster
```

### Step 4: Create Task Definition

Save as `task-definition.json`:
```json
{
  "family": "ai-resume-parser",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "ai-resume-parser",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/ai-resume-parser:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8001,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "LOG_LEVEL", "value": "info"},
        {"name": "HOST", "value": "0.0.0.0"},
        {"name": "PORT", "value": "8001"}
      ],
      "secrets": [
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:ai-parser-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-resume-parser",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8001/api/v1/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

Register it:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Step 5: Create Service with Load Balancer

```bash
aws ecs create-service \
    --cluster applymate-cluster \
    --service-name ai-resume-parser-service \
    --task-definition ai-resume-parser \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

---

## Option 4: AWS App Runner (Simplest)

```bash
aws apprunner create-service \
    --service-name ai-resume-parser \
    --source-configuration '{
        "AuthenticationConfiguration": {
            "AccessRoleArn": "arn:aws:iam::<account-id>:role/AppRunnerECRAccessRole"
        },
        "ImageRepository": {
            "ImageIdentifier": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/ai-resume-parser:latest",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8001",
                "RuntimeEnvironmentVariables": {
                    "API_KEY": "your-secure-api-key",
                    "LOG_LEVEL": "info"
                }
            }
        }
    }' \
    --instance-configuration '{
        "Cpu": "1024",
        "Memory": "2048"
    }'
```

---

## Cost Comparison (Approximate Monthly)

| Option | Cost | Best For |
|--------|------|----------|
| EC2 t3.small | ~$15/month | Development, low traffic |
| EC2 t3.medium | ~$30/month | Production, moderate traffic |
| ECS Fargate | ~$25-50/month | Auto-scaling, managed |
| App Runner | ~$25-40/month | Simplest setup |

---

## HTTPS Setup (Recommended for Production)

### Using Nginx as Reverse Proxy

```bash
# Install nginx
sudo yum install nginx -y  # Amazon Linux
sudo apt install nginx -y   # Ubuntu

# Configure
sudo nano /etc/nginx/conf.d/ai-parser.conf
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Add SSL with Certbot

```bash
sudo yum install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## Monitoring & Logs

```bash
# View container logs
docker logs -f ai-resume-parser

# Check container status
docker ps
docker stats ai-resume-parser

# CloudWatch (if using ECS)
aws logs tail /ecs/ai-resume-parser --follow
```

---

## Troubleshooting

### Container won't start
```bash
docker logs ai-resume-parser
docker inspect ai-resume-parser
```

### Out of memory
- Upgrade to t3.medium (4GB RAM)
- Or add swap space on t3.small

### Health check failing
- Wait 60+ seconds after start (spaCy model loading)
- Check if port 8001 is accessible

### Connection refused from Laravel
- Check Security Group allows inbound on port 8001
- Verify EC2 public IP in Laravel .env
- Test: `curl http://<ec2-ip>:8001/api/v1/health`
