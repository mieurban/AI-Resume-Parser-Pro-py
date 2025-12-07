from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import sys
import re
import uuid
from datetime import datetime
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.file_processor import FileProcessor
from core.nlp_engine import NlpEngine
from core.ml_models import MLModels

# Configuration
API_KEY = os.getenv("API_KEY", "dev-api-key-12345")

app = FastAPI(
    title="AI-Powered Resume Parser API",
    description="World-class resume parsing system with NLP and ML capabilities",
    version="1.0.0",
    openapi_tags=[
        {"name": "parsing", "description": "Resume parsing operations"},
        {"name": "matching", "description": "Job matching operations"},
        {"name": "skills", "description": "Skill operations"},
        {"name": "system", "description": "System operations"},
    ]
)

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

# Track app state
app_state = {
    "start_time": time.time(),
    "models_loaded": False,
    "last_parse_time_ms": 0,
    "total_parses": 0,
}

# Initialize components
_file_processor = None
_nlp_engine = None
_ml_models = None


def init_components():
    """Initialize components at startup."""
    global _file_processor, _nlp_engine, _ml_models, app_state
    
    print("Initializing components...")
    
    try:
        _file_processor = FileProcessor()
        print("✓ FileProcessor loaded")
    except Exception as e:
        print(f"✗ Failed to load FileProcessor: {e}")
    
    try:
        _nlp_engine = NlpEngine()
        app_state["models_loaded"] = True
        model_name = getattr(_nlp_engine, 'model_name', 'unknown')
        print(f"✓ NlpEngine loaded with model: {model_name}")
    except Exception as e:
        print(f"✗ Failed to load NLP engine: {e}")
        app_state["models_loaded"] = False
    
    try:
        _ml_models = MLModels()
        print("✓ MLModels loaded")
    except Exception as e:
        print(f"✗ Failed to load ML models: {e}")
    
    print(f"Models loaded: {app_state['models_loaded']}")


def get_components():
    """Get initialized components."""
    global _file_processor, _nlp_engine, _ml_models
    return _file_processor, _nlp_engine, _ml_models


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    init_components()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify API key from Authorization header."""
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


# ============================================================
# Request/Response Models
# ============================================================
class JobDescription(BaseModel):
    title: str
    description: str
    requirements: List[str]
    preferred_qualifications: Optional[List[str]] = None


class ParseTextRequest(BaseModel):
    text: str
    options: Optional[Dict[str, Any]] = None


class NormalizeSkillsRequest(BaseModel):
    skills: List[str]


class JobMatchRequest(BaseModel):
    resume_data: Dict[str, Any]
    job_description: Dict[str, Any]


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
    """Extract current role from entities."""
    experience = entities.get("experience", [])
    if experience and len(experience) > 0:
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
@app.get("/api/v1/health", tags=["system"])
async def health_check():
    """Health check endpoint - no authentication required."""
    uptime = time.time() - app_state["start_time"]
    
    return {
        "status": "healthy" if app_state["models_loaded"] else "degraded",
        "version": "1.0.0",
        "models_loaded": app_state["models_loaded"],
        "uptime_seconds": int(uptime),
        "memory_usage_mb": get_memory_usage(),
        "last_parse_time_ms": app_state["last_parse_time_ms"],
        "total_parses": app_state["total_parses"],
    }


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
        }
    }


# ============================================================
# Parsing endpoints
# ============================================================
@app.post("/api/v1/parse-resume", tags=["parsing"])
async def parse_resume_v1(
    file: UploadFile = File(...),
    authorized: bool = Depends(verify_api_key)
):
    """Parse resume file and extract structured data."""
    global app_state
    start_time = time.time()
    
    try:
        fp, nlp, ml = get_components()
        
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
                "certifications": entities.get("certifications", []),
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
                "sbert": None,
            }
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/parse-text", tags=["parsing"])
async def parse_text_v1(
    request: ParseTextRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Parse resume text and extract structured data."""
    global app_state
    start_time = time.time()
    
    try:
        fp, nlp, ml = get_components()
        
        text = request.text
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
                "certifications": entities.get("certifications", []),
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Job matching endpoints
# ============================================================
@app.post("/api/v1/match-job", tags=["matching"])
async def match_job_v1(
    request: JobMatchRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Match resume data against job description."""
    try:
        fp, nlp, ml = get_components()
        
        resume_data = request.resume_data
        job_description = request.job_description
        
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Skills endpoints
# ============================================================
@app.post("/api/v1/normalize-skills", tags=["skills"])
async def normalize_skills_v1(
    request: NormalizeSkillsRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Normalize skills to standard taxonomy."""
    try:
        fp, nlp, ml = get_components()
        
        skills = request.skills
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
        raise HTTPException(status_code=500, detail=str(e))


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
