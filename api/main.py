from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import os
import sys
import re
import uuid
from datetime import datetime
import json
import time
import logging
from pythonjsonlogger import jsonlogger

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.file_processor import FileProcessor
from core.nlp_engine import NlpEngine
from core.ml_models import MLModels

# ============================================================
# Logging Configuration
# ============================================================
def setup_logging():
    """Configure structured JSON logging."""
    log_logger = logging.getLogger()
    log_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    log_logger.handlers = []
    
    # JSON formatter for structured logging
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        rename_fields={'asctime': 'timestamp', 'levelname': 'level'}
    )
    handler.setFormatter(formatter)
    log_logger.addHandler(handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Configuration
API_KEY = os.getenv("API_KEY", "dev-api-key-12345")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize components
_file_processor = None
_nlp_engine = None
_ml_models = None

# Track app state
app_state = {
    "start_time": time.time(),
    "models_loaded": False,
    "last_parse_time_ms": 0,
    "total_parses": 0,
}


def init_components():
    """Initialize components at startup."""
    global _file_processor, _nlp_engine, _ml_models, app_state
    
    logger.info("Initializing components...")
    
    try:
        _file_processor = FileProcessor()
        logger.info("FileProcessor loaded", extra={"component": "file_processor", "status": "success"})
    except Exception as e:
        logger.error("Failed to load FileProcessor", extra={"component": "file_processor", "error": str(e)})
    
    try:
        _nlp_engine = NlpEngine()
        app_state["models_loaded"] = True
        model_name = getattr(_nlp_engine, 'model_name', 'unknown')
        logger.info("NlpEngine loaded", extra={"component": "nlp_engine", "model": model_name, "status": "success"})
    except Exception as e:
        logger.error("Failed to load NLP engine", extra={"component": "nlp_engine", "error": str(e)})
        app_state["models_loaded"] = False
    
    try:
        _ml_models = MLModels()
        logger.info("MLModels loaded", extra={"component": "ml_models", "status": "success"})
    except Exception as e:
        logger.error("Failed to load ML models", extra={"component": "ml_models", "error": str(e)})
    
    logger.info("Component initialization complete", extra={"models_loaded": app_state["models_loaded"]})


def get_components():
    """Get initialized components."""
    global _file_processor, _nlp_engine, _ml_models
    return _file_processor, _nlp_engine, _ml_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    init_components()
    logger.info("API startup complete")
    yield
    # Shutdown
    logger.info("API shutting down")


app = FastAPI(
    title="AI-Powered Resume Parser API",
    description="World-class resume parsing system with NLP and ML capabilities",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "parsing", "description": "Resume parsing operations"},
        {"name": "matching", "description": "Job matching operations"},
        {"name": "skills", "description": "Skill operations"},
        {"name": "system", "description": "System operations"},
    ]
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify API key from Authorization header."""
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


# ============================================================
# Custom Exception Handler
# ============================================================
class APIError(Exception):
    """Custom API error with structured response."""
    def __init__(self, message: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR", details: Dict = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handle custom API errors with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error("Unhandled exception", extra={"error": str(exc), "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc)} if os.getenv("DEBUG", "false").lower() == "true" else {},
            },
            "timestamp": datetime.now().isoformat(),
        }
    )


# ============================================================
# Request/Response Models
# ============================================================
class ContactInfo(BaseModel):
    """Contact information model."""
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None


class EducationEntry(BaseModel):
    """Education entry model."""
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[str] = None


class ExperienceEntry(BaseModel):
    """Work experience entry model."""
    company: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    responsibilities: Optional[List[str]] = None


class ConfidenceScores(BaseModel):
    """Confidence scores for extracted data."""
    name: float = Field(default=0.0, ge=0.0, le=1.0)
    contact: float = Field(default=0.0, ge=0.0, le=1.0)
    experience: float = Field(default=0.0, ge=0.0, le=1.0)
    skills: float = Field(default=0.0, ge=0.0, le=1.0)


class ParsedResumeData(BaseModel):
    """Parsed resume data model."""
    candidate_name: Optional[str] = None
    contact: Optional[ContactInfo] = None
    current_role: Optional[str] = None
    years_experience: Optional[int] = None
    education: Optional[List[EducationEntry]] = None
    experience: Optional[List[ExperienceEntry]] = None
    skills: Optional[List[str]] = None
    skills_normalized: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    summary: Optional[str] = None


class ParseResumeResponse(BaseModel):
    """Response model for resume parsing."""
    success: bool = True
    data: ParsedResumeData
    confidence_scores: ConfidenceScores
    processing_time_ms: int
    model_versions: Optional[Dict[str, Optional[str]]] = None


class CompatibilityScores(BaseModel):
    """Job matching compatibility scores."""
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    skill_match: float = Field(default=0.0, ge=0.0, le=1.0)
    experience_match: float = Field(default=0.0, ge=0.0, le=1.0)
    education_match: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class JobMatchResponse(BaseModel):
    """Response model for job matching."""
    success: bool = True
    compatibility: CompatibilityScores
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    recommendations: List[str] = []


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    models_loaded: bool
    uptime_seconds: int
    memory_usage_mb: float
    last_parse_time_ms: int
    total_parses: int


class JobDescription(BaseModel):
    """Job description input model."""
    title: str
    description: str
    requirements: List[str]
    preferred_qualifications: Optional[List[str]] = None


class ParseTextRequest(BaseModel):
    """Parse text request model."""
    text: str = Field(..., min_length=10, description="Resume text to parse")
    options: Optional[Dict[str, Any]] = None


class NormalizeSkillsRequest(BaseModel):
    """Normalize skills request model."""
    skills: List[str] = Field(..., min_items=1, description="List of skills to normalize")


class JobMatchRequest(BaseModel):
    """Job match request model."""
    resume_data: Dict[str, Any]
    job_description: Dict[str, Any]


class BatchParseRequest(BaseModel):
    """Batch parse request model."""
    texts: List[str] = Field(..., min_items=1, max_items=10, description="List of resume texts to parse")
    options: Optional[Dict[str, Any]] = None


# ============================================================
# Helper Functions
# ============================================================
def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return round(process.memory_info().rss / 1024 / 1024, 2)
    except ImportError:
        return 0.0


def extract_current_role(entities: Dict) -> Optional[str]:
    """Extract current role from entities - looks for job with 'Present' or no end date."""
    experience = entities.get("experience", [])
    if not experience:
        return None
    
    # First, try to find a job that is marked as current (end_date is "Present" or similar)
    current_indicators = ['present', 'current', 'now', 'ongoing', 'till date', 'to date']
    
    for exp in experience:
        if isinstance(exp, dict):
            end_date = exp.get("end_date", "") or ""
            end_date_lower = str(end_date).lower().strip()
            
            # Check if this is the current job
            is_current = (
                not end_date or  # No end date
                end_date_lower == "" or
                any(indicator in end_date_lower for indicator in current_indicators)
            )
            
            if is_current:
                title = exp.get("title") or exp.get("position")
                company = exp.get("company") or exp.get("organization")
                if title and company:
                    return f"{title} at {company}"
                return title or company
    
    # Fallback: return the first experience if no "current" job found
    first_exp = experience[0]
    if isinstance(first_exp, dict):
        title = first_exp.get("title") or first_exp.get("position")
        company = first_exp.get("company") or first_exp.get("organization")
        if title and company:
            return f"{title} at {company}"
        return title or company
    
    return None


def extract_years_experience(text: str) -> int:
    """Extract years of experience from text."""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'(?:experience|exp)\s*[:\-]?\s*(\d+)\+?\s*years?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def extract_languages(text: str) -> List[str]:
    """Extract spoken languages from text."""
    languages = []
    common_languages = [
        'English', 'Spanish', 'French', 'German', 'Chinese', 'Mandarin',
        'Japanese', 'Korean', 'Portuguese', 'Italian', 'Russian', 'Arabic',
        'Hindi', 'Dutch', 'Swedish', 'Polish', 'Turkish', 'Vietnamese',
    ]
    for lang in common_languages:
        if re.search(rf'\b{lang}\b', text, re.IGNORECASE):
            languages.append(lang)
    return languages[:5]


def extract_summary(text: str) -> Optional[str]:
    """Extract summary from text."""
    # Look for summary/objective section
    patterns = [
        r'(?:summary|objective|profile|about)\s*[:\-]?\s*(.{50,300})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # Return first 200 chars as fallback
    if len(text) > 50:
        return text[:200].strip() + "..."
    return None


def generate_recommendations(missing_skills: List[str]) -> List[str]:
    """Generate recommendations based on missing skills."""
    recommendations = []
    if missing_skills:
        recommendations.append(f"Consider adding these skills to your profile: {', '.join(missing_skills[:3])}")
        if len(missing_skills) > 3:
            recommendations.append(f"Focus on developing expertise in {missing_skills[0]}")
    return recommendations


def normalize_certifications(certifications: List) -> List[str]:
    """Convert certifications from dict format to list of strings and deduplicate."""
    if not certifications:
        return []
    
    result = []
    for cert in certifications:
        if isinstance(cert, str):
            name = cert.strip()
        elif isinstance(cert, dict):
            name = cert.get('name', '').strip()
        else:
            name = str(cert).strip()
        
        if name:
            # Clean up newlines and extra whitespace
            name = ' '.join(name.split())
            result.append(name)
    
    # Deduplicate: remove entries that are substrings of other entries
    # or are very similar (case-insensitive)
    unique = []
    seen_lower = set()
    
    # Sort by length descending so longer (more complete) names come first
    result_sorted = sorted(result, key=len, reverse=True)
    
    for name in result_sorted:
        name_lower = name.lower()
        
        # Skip if exact match already seen
        if name_lower in seen_lower:
            continue
        
        # Skip if this is a substring of an already-added certification
        is_substring = False
        for existing in unique:
            existing_lower = existing.lower()
            if name_lower in existing_lower or existing_lower in name_lower:
                is_substring = True
                break
        
        if not is_substring:
            unique.append(name)
            seen_lower.add(name_lower)
    
    return unique


def extract_basic_entities(text: str) -> Dict[str, Any]:
    """Basic entity extraction without NLP models."""
    # Email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    email = email_match.group() if email_match else None
    
    # Phone - multiple patterns
    phone_patterns = [
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{5}[-.\s]?\d{5}',
    ]
    phone = None
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match:
            phone = phone_match.group()
            break
    
    # LinkedIn
    linkedin_match = re.search(r'linkedin\.com/in/([a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    linkedin = f"https://linkedin.com/in/{linkedin_match.group(1)}" if linkedin_match else None
    
    # GitHub
    github_match = re.search(r'github\.com/([a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    github = f"https://github.com/{github_match.group(1)}" if github_match else None
    
    # Website (exclude social media)
    website = None
    website_match = re.search(r'https?://(?:www\.)?([a-zA-Z0-9-]+\.(?:com|io|dev|me|co|org|net))', text)
    if website_match:
        site = website_match.group(0)
        if not any(skip in site.lower() for skip in ['linkedin', 'github', 'twitter', 'facebook']):
            website = site
    
    return {
        "name": None,
        "contact": {
            "email": email, 
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
            "website": website,
            "location": None
        },
        "education": [],
        "experience": [],
        "skills": [],
        "certifications": [],
    }


# ============================================================
# Health endpoint (no auth required)
# ============================================================
@app.get("/api/v1/health", tags=["system"], response_model=HealthResponse)
async def health_check():
    """Health check endpoint - no authentication required."""
    uptime = time.time() - app_state["start_time"]
    
    return HealthResponse(
        status="healthy" if app_state["models_loaded"] else "degraded",
        version="1.0.0",
        models_loaded=app_state["models_loaded"],
        uptime_seconds=int(uptime),
        memory_usage_mb=get_memory_usage(),
        last_parse_time_ms=app_state["last_parse_time_ms"],
        total_parses=app_state["total_parses"],
    )


@app.get("/api/v1/models/status", tags=["system"])
async def models_status(authorized: bool = Depends(verify_api_key)):
    """Get ML models status."""
    model_name = None
    fp, nlp, ml = get_components()
    if app_state["models_loaded"] and nlp:
        model_name = getattr(nlp, 'model_name', 'en_core_web_sm')
    return {
        "spacy": {
            "loaded": app_state["models_loaded"],
            "model": model_name,
        },
        "nltk": {
            "loaded": True,
        },
        "sbert": {
            "loaded": ml is not None,
            "model": "all-MiniLM-L6-v2" if ml else None,
        }
    }


# ============================================================
# Parsing endpoints
# ============================================================
@app.post("/api/v1/parse-resume", tags=["parsing"], response_model=ParseResumeResponse)
@limiter.limit(RATE_LIMIT)
async def parse_resume_v1(
    request: Request,
    file: UploadFile = File(...),
    authorized: bool = Depends(verify_api_key)
):
    """Parse resume file and extract structured data."""
    global app_state
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info("Parse resume request started", extra={
        "request_id": request_id,
        "file_name": file.filename,
        "content_type": file.content_type
    })
    
    try:
        fp, nlp, ml = get_components()
        
        if fp is None:
            raise APIError("File processor not available", status_code=503, error_code="SERVICE_UNAVAILABLE")
        
        # Get file extension from filename or content type
        file_ext = os.path.splitext(file.filename or '')[1].lower()
        
        # If no extension, try to determine from content type
        if not file_ext:
            content_type = file.content_type or ''
            content_type_map = {
                'application/pdf': '.pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'application/msword': '.doc',
                'text/plain': '.txt',
                'image/png': '.png',
                'image/jpeg': '.jpg',
            }
            file_ext = content_type_map.get(content_type, '.pdf')  # Default to PDF
        
        temp_file = f"temp_{uuid.uuid4()}{file_ext}"
        
        with open(temp_file, "wb") as buffer:
            buffer.write(file.file.read())
            
        # Process file
        text = fp.extract_text(temp_file)
        entities = nlp.extract_entities(text) if nlp else extract_basic_entities(text)
        
        # Normalize skills
        if 'skills' in entities and ml:
            entities['skills_normalized'] = ml.normalize_skills(entities['skills'])
        else:
            entities['skills_normalized'] = entities.get('skills', [])
        
        # Clean up
        os.remove(temp_file)
        
        # Update stats
        processing_time = int((time.time() - start_time) * 1000)
        app_state["last_parse_time_ms"] = processing_time
        app_state["total_parses"] += 1
        
        logger.info("Parse resume completed", extra={
            "request_id": request_id,
            "processing_time_ms": processing_time,
            "candidate_name": entities.get("name"),
            "skills_count": len(entities.get("skills", []))
        })
        
        return {
            "success": True,
            "data": {
                "candidate_name": entities.get("name"),
                "contact": entities.get("contact", {}),
                "current_role": extract_current_role(entities),
                "years_experience": extract_years_experience(text),
                "education": entities.get("education", []),
                "experience": entities.get("experience", []),
                "skills": entities.get("skills", []),
                "skills_normalized": entities.get("skills_normalized", []),
                "certifications": normalize_certifications(entities.get("certifications", [])),
                "languages": extract_languages(text),
                "summary": extract_summary(text),
            },
            "confidence_scores": {
                "name": 0.9 if entities.get("name") else 0.0,
                "contact": 0.9 if entities.get("contact", {}).get("email") else 0.5,
                "experience": 0.85,
                "skills": 0.9 if entities.get("skills") else 0.0,
            },
            "processing_time_ms": processing_time,
            "model_versions": {
                "spacy": getattr(nlp, 'model_name', 'en_core_web_sm') if nlp else 'en_core_web_sm',
                "sbert": "all-MiniLM-L6-v2" if ml else None,
            }
        }
        
    except APIError:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        logger.error("Parse resume failed", extra={"request_id": request_id, "error": str(e)})
        raise APIError(
            message=f"Failed to parse resume: {str(e)}",
            status_code=500,
            error_code="PARSE_FAILED"
        )


@app.post("/api/v1/parse-text", tags=["parsing"], response_model=ParseResumeResponse)
@limiter.limit(RATE_LIMIT)
async def parse_text_v1(
    request: Request,
    body: ParseTextRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Parse resume text and extract structured data."""
    global app_state
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info("Parse text request started", extra={
        "request_id": request_id,
        "text_length": len(body.text)
    })
    
    try:
        fp, nlp, ml = get_components()
        
        text = body.text
        entities = nlp.extract_entities(text) if nlp else extract_basic_entities(text)
        
        # Normalize skills
        if 'skills' in entities and ml:
            entities['skills_normalized'] = ml.normalize_skills(entities['skills'])
        else:
            entities['skills_normalized'] = entities.get('skills', [])
        
        # Update stats
        processing_time = int((time.time() - start_time) * 1000)
        app_state["last_parse_time_ms"] = processing_time
        app_state["total_parses"] += 1
        
        return {
            "success": True,
            "data": {
                "candidate_name": entities.get("name"),
                "contact": entities.get("contact", {}),
                "current_role": extract_current_role(entities),
                "years_experience": extract_years_experience(text),
                "education": entities.get("education", []),
                "experience": entities.get("experience", []),
                "skills": entities.get("skills", []),
                "skills_normalized": entities.get("skills_normalized", []),
                "certifications": normalize_certifications(entities.get("certifications", [])),
                "languages": extract_languages(text),
                "summary": extract_summary(text),
            },
            "confidence_scores": {
                "name": 0.9 if entities.get("name") else 0.0,
                "contact": 0.9 if entities.get("contact", {}).get("email") else 0.5,
                "experience": 0.85,
                "skills": 0.9 if entities.get("skills") else 0.0,
            },
            "processing_time_ms": processing_time,
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error("Parse text failed", extra={"request_id": request_id, "error": str(e)})
        raise APIError(
            message=f"Failed to parse text: {str(e)}",
            status_code=500,
            error_code="PARSE_FAILED"
        )


# ============================================================
# Batch parsing endpoint
# ============================================================
@app.post("/api/v1/parse-batch", tags=["parsing"])
@limiter.limit("20/minute")
async def parse_batch_v1(
    request: Request,
    body: BatchParseRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Parse multiple resume texts in a single request (max 10)."""
    global app_state
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info("Batch parse request started", extra={
        "request_id": request_id,
        "batch_size": len(body.texts)
    })
    
    results = []
    fp, nlp, ml = get_components()
    
    for idx, text in enumerate(body.texts):
        try:
            entities = nlp.extract_entities(text) if nlp else extract_basic_entities(text)
            
            if 'skills' in entities and ml:
                entities['skills_normalized'] = ml.normalize_skills(entities['skills'])
            else:
                entities['skills_normalized'] = entities.get('skills', [])
            
            results.append({
                "index": idx,
                "success": True,
                "data": {
                    "candidate_name": entities.get("name"),
                    "contact": entities.get("contact", {}),
                    "current_role": extract_current_role(entities),
                    "years_experience": extract_years_experience(text),
                    "education": entities.get("education", []),
                    "experience": entities.get("experience", []),
                    "skills": entities.get("skills", []),
                    "skills_normalized": entities.get("skills_normalized", []),
                    "certifications": normalize_certifications(entities.get("certifications", [])),
                    "languages": extract_languages(text),
                    "summary": extract_summary(text),
                }
            })
        except Exception as e:
            results.append({
                "index": idx,
                "success": False,
                "error": str(e)
            })
    
    processing_time = int((time.time() - start_time) * 1000)
    app_state["total_parses"] += len([r for r in results if r["success"]])
    
    logger.info("Batch parse completed", extra={
        "request_id": request_id,
        "processing_time_ms": processing_time,
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]])
    })
    
    return {
        "success": True,
        "results": results,
        "processing_time_ms": processing_time,
        "summary": {
            "total": len(body.texts),
            "successful": len([r for r in results if r["success"]]),
            "failed": len([r for r in results if not r["success"]])
        }
    }


# ============================================================
# Job matching endpoints
# ============================================================
@app.post("/api/v1/match-job", tags=["matching"], response_model=JobMatchResponse)
@limiter.limit(RATE_LIMIT)
async def match_job_v1(
    request: Request,
    body: JobMatchRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Match resume data against job description."""
    try:
        fp, nlp, ml = get_components()
        
        resume_data = body.resume_data
        job_description = body.job_description
        
        # Normalize skills if not already done
        if 'skills' in resume_data and ml:
            resume_data['skills'] = ml.normalize_skills(resume_data['skills'])
            
        compatibility = ml.calculate_compatibility(resume_data, job_description) if ml else {}
        
        # Extract matched and missing skills
        resume_skills = set(s.lower() for s in resume_data.get('skills', []))
        job_requirements = set(r.lower() for r in job_description.get('requirements', []))
        
        matched_skills = list(resume_skills & job_requirements)
        missing_skills = list(job_requirements - resume_skills)
        
        return {
            "success": True,
            "compatibility": {
                "overall_score": compatibility.get("overall_score", 0.0),
                "skill_match": compatibility.get("skill_match", 0.0),
                "experience_match": compatibility.get("tfidf_similarity", 0.0),
                "education_match": 0.0,
                "semantic_similarity": compatibility.get("semantic_similarity", 0.0),
            },
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "recommendations": generate_recommendations(missing_skills),
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error("Job matching failed", extra={"error": str(e)})
        raise APIError(
            message=f"Job matching failed: {str(e)}",
            status_code=500,
            error_code="MATCH_FAILED"
        )


# ============================================================
# Skills endpoints
# ============================================================
@app.post("/api/v1/normalize-skills", tags=["skills"])
@limiter.limit(RATE_LIMIT)
async def normalize_skills_v1(
    request: Request,
    body: NormalizeSkillsRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Normalize skills to standard taxonomy."""
    try:
        fp, nlp, ml = get_components()
        
        skills = body.skills
        normalized = ml.normalize_skills(skills) if ml else skills
        
        # Create mappings
        mappings = {}
        for i, skill in enumerate(skills):
            if i < len(normalized):
                mappings[skill] = normalized[i]
        
        return {
            "success": True,
            "normalized": normalized,
            "mappings": mappings,
        }
        
    except Exception as e:
        logger.error("Skill normalization failed", extra={"error": str(e)})
        raise APIError(
            message=f"Skill normalization failed: {str(e)}",
            status_code=500,
            error_code="NORMALIZE_FAILED"
        )


# ============================================================
# Legacy endpoints (for backward compatibility)
# ============================================================
@app.post("/parse-resume", tags=["parsing"], include_in_schema=False)
async def parse_resume_legacy(
    file: UploadFile = File(...), 
    job_description: Optional[str] = None
):
    """Legacy endpoint - Parse resume and optionally match against job description"""
    try:
        fp, nlp, ml = get_components()
        
        # Save uploaded file temporarily
        file_ext = os.path.splitext(file.filename)[1]
        temp_file = f"temp_{uuid.uuid4()}{file_ext}"
        
        with open(temp_file, "wb") as buffer:
            buffer.write(file.file.read())
            
        # Process file
        text = fp.extract_text(temp_file)
        entities = nlp.extract_entities(text) if nlp else extract_basic_entities(text)
        
        # Normalize skills
        if 'skills' in entities and ml:
            entities['skills'] = ml.normalize_skills(entities['skills'])
            
        # Calculate compatibility if job description provided
        compatibility = None
        if job_description:
            try:
                job_data = json.loads(job_description)
            except json.JSONDecodeError:
                job_data = {"description": job_description}
                
            compatibility = ml.calculate_compatibility(entities, job_data) if ml else None
        
        # Clean up
        os.remove(temp_file)
        
        return {
            "success": True,
            "data": entities,
            "compatibility": compatibility,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/match-resume", tags=["matching"], include_in_schema=False)
async def match_resume_legacy(
    resume_data: Dict[str, Any], 
    job_description: JobDescription
):
    """Legacy endpoint - Match existing resume data against job description"""
    try:
        fp, nlp, ml = get_components()
        
        # Normalize skills if not already done
        if 'skills' in resume_data and ml:
            resume_data['skills'] = ml.normalize_skills(resume_data['skills'])
            
        compatibility = ml.calculate_compatibility(resume_data, job_description.dict()) if ml else {}
        
        return {
            "success": True,
            "compatibility": compatibility,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["system"], include_in_schema=False)
async def health_check_legacy():
    """Legacy health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
