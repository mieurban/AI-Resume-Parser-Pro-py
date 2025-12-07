#!/bin/bash

# =============================================================================
# AI Resume Parser API - Production Setup Script
# =============================================================================
# This script sets up the AI Resume Parser API for production use.
# It installs all dependencies, downloads required models, and configures
# the environment for deployment.
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
PYTHON_VERSION="3.11"
PORT="${PORT:-8001}"
API_KEY="${API_KEY:-$(openssl rand -hex 32)}"

# Functions
print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# =============================================================================
# Pre-flight Checks
# =============================================================================
print_header "Pre-flight Checks"

# Check Python version
echo "Checking Python installation..."
if command -v python${PYTHON_VERSION} &> /dev/null; then
    PYTHON_CMD="python${PYTHON_VERSION}"
    print_success "Python ${PYTHON_VERSION} found"
elif command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PY_VER" == "3.11" || "$PY_VER" == "3.10" || "$PY_VER" == "3.12" ]]; then
        PYTHON_CMD="python3"
        print_success "Python $PY_VER found"
    else
        print_warning "Python $PY_VER found. Python 3.10-3.12 recommended for best compatibility."
        PYTHON_CMD="python3"
    fi
else
    print_error "Python 3 not found. Please install Python 3.10 or higher."
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# Check pip
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip not found for $PYTHON_CMD"
    exit 1
fi
print_success "pip is available"

# Check for Tesseract (optional, for OCR)
if command -v tesseract &> /dev/null; then
    print_success "Tesseract OCR is installed (optional)"
else
    print_warning "Tesseract OCR not found (optional - needed for image-based PDFs)"
    echo "  macOS: brew install tesseract"
    echo "  Ubuntu: sudo apt install tesseract-ocr"
fi

# =============================================================================
# Setup Virtual Environment
# =============================================================================
print_header "Setting Up Virtual Environment"

cd "$SCRIPT_DIR"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists. Recreating..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment..."
$PYTHON_CMD -m venv "$VENV_DIR"
print_success "Virtual environment created"

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools > /dev/null 2>&1
print_success "pip upgraded"

# =============================================================================
# Install Dependencies
# =============================================================================
print_header "Installing Dependencies"

echo "Installing Python packages (this may take a few minutes)..."

# Install main dependencies
pip install -r requirements.txt 2>&1 | while read line; do
    if [[ $line == *"Successfully installed"* ]]; then
        echo -e "${GREEN}✓${NC} Packages installed"
    elif [[ $line == *"ERROR"* ]] || [[ $line == *"error"* ]]; then
        echo -e "${RED}$line${NC}"
    fi
done

print_success "Python dependencies installed"

# =============================================================================
# Download NLP Models
# =============================================================================
print_header "Downloading NLP Models"

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm 2>&1 | tail -1
print_success "spaCy model (en_core_web_sm) downloaded"

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
print('NLTK data downloaded successfully')
"
print_success "NLTK data downloaded"

# =============================================================================
# Configure Environment
# =============================================================================
print_header "Configuring Environment"

ENV_FILE="${SCRIPT_DIR}/.env"

if [ -f "$ENV_FILE" ]; then
    print_warning ".env file already exists. Backing up to .env.backup"
    cp "$ENV_FILE" "${ENV_FILE}.backup"
fi

# Generate new .env file
cat > "$ENV_FILE" << EOF
# AI Resume Parser API Configuration
# Generated on $(date)

# API Security Key (keep this secret!)
API_KEY=${API_KEY}

# Server Configuration
PORT=${PORT}
HOST=0.0.0.0

# Environment (development/production)
ENVIRONMENT=production

# Logging
LOG_LEVEL=INFO

# CORS (comma-separated origins, or * for all)
CORS_ORIGINS=*

# Model Configuration
SPACY_MODEL=en_core_web_sm

# Performance
WORKERS=4
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50
TIMEOUT=120
GRACEFUL_TIMEOUT=30
EOF

print_success "Environment file created"
echo -e "  API Key: ${YELLOW}${API_KEY}${NC}"
echo -e "  Port: ${PORT}"

# =============================================================================
# Create Systemd Service (Linux only)
# =============================================================================
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_header "Creating Systemd Service"
    
    SERVICE_FILE="/tmp/ai-resume-parser.service"
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=AI Resume Parser API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=${SCRIPT_DIR}
Environment="PATH=${VENV_DIR}/bin"
ExecStart=${VENV_DIR}/bin/uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    print_success "Systemd service file created at: $SERVICE_FILE"
    echo ""
    echo "To install the service, run:"
    echo "  sudo cp $SERVICE_FILE /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable ai-resume-parser"
    echo "  sudo systemctl start ai-resume-parser"
fi

# =============================================================================
# Create Run Scripts
# =============================================================================
print_header "Creating Run Scripts"

# Development script
cat > "${SCRIPT_DIR}/run-dev.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"
source "${SCRIPT_DIR}/.env" 2>/dev/null || true
uvicorn api.main:app --reload --host 0.0.0.0 --port ${PORT:-8001}
EOF
chmod +x "${SCRIPT_DIR}/run-dev.sh"
print_success "Development script created: run-dev.sh"

# Production script
cat > "${SCRIPT_DIR}/run-prod.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"
source "${SCRIPT_DIR}/.env" 2>/dev/null || true

# Production settings
WORKERS=${WORKERS:-4}
PORT=${PORT:-8001}
HOST=${HOST:-0.0.0.0}

echo "Starting AI Resume Parser API in production mode..."
echo "  Workers: $WORKERS"
echo "  Port: $PORT"
echo "  Host: $HOST"

exec uvicorn api.main:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --timeout-keep-alive 30 \
    --access-log \
    --log-level info
EOF
chmod +x "${SCRIPT_DIR}/run-prod.sh"
print_success "Production script created: run-prod.sh"

# Health check script
cat > "${SCRIPT_DIR}/healthcheck.sh" << 'EOF'
#!/bin/bash
PORT=${PORT:-8001}
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/api/v1/health)
if [ "$response" = "200" ]; then
    echo "✓ API is healthy"
    curl -s http://localhost:$PORT/api/v1/health | python3 -m json.tool
    exit 0
else
    echo "✗ API is not responding (HTTP $response)"
    exit 1
fi
EOF
chmod +x "${SCRIPT_DIR}/healthcheck.sh"
print_success "Health check script created: healthcheck.sh"

# =============================================================================
# Verify Installation
# =============================================================================
print_header "Verifying Installation"

echo "Testing Python imports..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import fastapi
    print(f'✓ FastAPI {fastapi.__version__}')
except ImportError as e:
    print(f'✗ FastAPI: {e}')

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print(f'✓ spaCy {spacy.__version__} (model: en_core_web_sm)')
except Exception as e:
    print(f'✗ spaCy: {e}')

try:
    import nltk
    print(f'✓ NLTK {nltk.__version__}')
except ImportError as e:
    print(f'✗ NLTK: {e}')

try:
    from pdfminer.high_level import extract_text
    print('✓ pdfminer.six')
except ImportError as e:
    print(f'✗ pdfminer: {e}')

try:
    from docx import Document
    print('✓ python-docx')
except ImportError as e:
    print(f'✗ python-docx: {e}')

print()
print('All core dependencies verified!')
"

# =============================================================================
# Summary
# =============================================================================
print_header "Setup Complete!"

echo -e "${GREEN}The AI Resume Parser API is ready for production!${NC}"
echo ""
echo "Configuration:"
echo "  API Key: ${API_KEY}"
echo "  Port: ${PORT}"
echo "  Directory: ${SCRIPT_DIR}"
echo ""
echo "Quick Start:"
echo "  Development: ./run-dev.sh"
echo "  Production:  ./run-prod.sh"
echo "  Health:      ./healthcheck.sh"
echo ""
echo "API Endpoints:"
echo "  Health:      http://localhost:${PORT}/api/v1/health"
echo "  Swagger:     http://localhost:${PORT}/docs"
echo "  Parse:       POST http://localhost:${PORT}/api/v1/parse-resume"
echo ""
echo "Laravel Integration (.env):"
echo "  AI_PARSER_PRO_ENABLED=true"
echo "  AI_PARSER_PRO_URL=http://localhost:${PORT}"
echo "  AI_PARSER_PRO_API_KEY=${API_KEY}"
echo ""
echo -e "${YELLOW}Don't forget to save your API key securely!${NC}"
