#!/bin/bash
# AWS ECS Fargate Deployment Script for AI Resume Parser API
# This script deploys the API to AWS ECS with Fargate

set -e

# ============================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="applymate-cluster"
SERVICE_NAME="ai-resume-parser-service"
TASK_FAMILY="ai-resume-parser"
ECR_REPO_NAME="ai-resume-parser"
API_KEY="${API_KEY:-change-this-to-secure-key}"

# ============================================
# DERIVED VALUES
# ============================================
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

echo "============================================"
echo "AWS ECS Deployment for AI Resume Parser API"
echo "============================================"
echo "Account ID: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo "ECR URI: $ECR_URI"
echo ""

# ============================================
# STEP 1: Create ECR Repository (if not exists)
# ============================================
echo "Step 1: Creating ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION 2>/dev/null || \
  aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION

# ============================================
# STEP 2: Build and Push Docker Image
# ============================================
echo "Step 2: Building and pushing Docker image..."

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build image
docker build -t $ECR_REPO_NAME:latest .

# Tag and push
docker tag $ECR_REPO_NAME:latest $ECR_URI:latest
docker push $ECR_URI:latest

echo "✅ Image pushed to ECR"

# ============================================
# STEP 3: Create ECS Cluster (if not exists)
# ============================================
echo "Step 3: Creating ECS cluster..."
aws ecs describe-clusters --clusters $CLUSTER_NAME --region $AWS_REGION | grep -q "ACTIVE" || \
  aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $AWS_REGION

echo "✅ Cluster ready"

# ============================================
# STEP 4: Create CloudWatch Log Group
# ============================================
echo "Step 4: Creating CloudWatch log group..."
aws logs create-log-group --log-group-name /ecs/ai-resume-parser --region $AWS_REGION 2>/dev/null || true

# ============================================
# STEP 5: Create/Update Task Definition
# ============================================
echo "Step 5: Registering task definition..."

# Generate task definition with actual values
cat > /tmp/task-definition.json << EOF
{
  "family": "$TASK_FAMILY",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "ai-resume-parser",
      "image": "$ECR_URI:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8001,
          "protocol": "tcp",
          "hostPort": 8001
        }
      ],
      "environment": [
        {"name": "LOG_LEVEL", "value": "info"},
        {"name": "HOST", "value": "0.0.0.0"},
        {"name": "PORT", "value": "8001"},
        {"name": "API_KEY", "value": "$API_KEY"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-resume-parser",
          "awslogs-region": "$AWS_REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8001/api/v1/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 120
      }
    }
  ]
}
EOF

aws ecs register-task-definition \
  --cli-input-json file:///tmp/task-definition.json \
  --region $AWS_REGION

echo "✅ Task definition registered"

# ============================================
# STEP 6: Get Default VPC and Subnets
# ============================================
echo "Step 6: Getting VPC configuration..."

# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $AWS_REGION)
echo "VPC ID: $VPC_ID"

# Get subnets
SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text --region $AWS_REGION | tr '\t' ',')
FIRST_SUBNET=$(echo $SUBNET_IDS | cut -d',' -f1)
echo "Using subnet: $FIRST_SUBNET"

# ============================================
# STEP 7: Create Security Group
# ============================================
echo "Step 7: Creating security group..."

# Check if security group exists
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=ai-resume-parser-sg" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text --region $AWS_REGION 2>/dev/null)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name ai-resume-parser-sg \
    --description "Security group for AI Resume Parser API" \
    --vpc-id $VPC_ID \
    --query "GroupId" --output text --region $AWS_REGION)
  
  # Allow inbound on port 8001
  aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8001 \
    --cidr 0.0.0.0/0 \
    --region $AWS_REGION
fi

echo "Security Group ID: $SG_ID"

# ============================================
# STEP 8: Create or Update ECS Service
# ============================================
echo "Step 8: Creating/updating ECS service..."

# Check if service exists
SERVICE_EXISTS=$(aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION --query "services[?status=='ACTIVE'].serviceName" --output text)

if [ -z "$SERVICE_EXISTS" ]; then
  echo "Creating new service..."
  aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_FAMILY \
    --desired-count 1 \
    --launch-type FARGATE \
    --platform-version LATEST \
    --network-configuration "awsvpcConfiguration={subnets=[$FIRST_SUBNET],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region $AWS_REGION
else
  echo "Updating existing service..."
  aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --task-definition $TASK_FAMILY \
    --force-new-deployment \
    --region $AWS_REGION
fi

echo "✅ Service deployed"

# ============================================
# STEP 9: Wait for Service to be Stable
# ============================================
echo "Step 9: Waiting for service to stabilize (this may take 2-3 minutes)..."
aws ecs wait services-stable --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION

# ============================================
# STEP 10: Get Public IP
# ============================================
echo "Step 10: Getting public IP..."

TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME --query "taskArns[0]" --output text --region $AWS_REGION)

if [ "$TASK_ARN" != "None" ] && [ -n "$TASK_ARN" ]; then
  ENI_ID=$(aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text --region $AWS_REGION)
  
  if [ "$ENI_ID" != "None" ] && [ -n "$ENI_ID" ]; then
    PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --query "NetworkInterfaces[0].Association.PublicIp" --output text --region $AWS_REGION)
    
    echo ""
    echo "============================================"
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo "============================================"
    echo ""
    echo "API Endpoint: http://$PUBLIC_IP:8001"
    echo "Health Check: http://$PUBLIC_IP:8001/api/v1/health"
    echo ""
    echo "Update your Laravel .env:"
    echo "AI_PARSER_PRO_URL=http://$PUBLIC_IP:8001"
    echo "AI_PARSER_PRO_API_KEY=$API_KEY"
    echo ""
  fi
fi

echo "============================================"
echo "Useful Commands:"
echo "============================================"
echo "View logs:    aws logs tail /ecs/ai-resume-parser --follow --region $AWS_REGION"
echo "Service info: aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION"
echo "Stop service: aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0 --region $AWS_REGION"
echo "Delete all:   See cleanup commands below"
echo ""
echo "============================================"
echo "Cleanup Commands (when you want to remove everything):"
echo "============================================"
echo "aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count 0 --region $AWS_REGION"
echo "aws ecs delete-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --region $AWS_REGION"
echo "aws ecs delete-cluster --cluster $CLUSTER_NAME --region $AWS_REGION"
echo "aws ecr delete-repository --repository-name $ECR_REPO_NAME --force --region $AWS_REGION"
