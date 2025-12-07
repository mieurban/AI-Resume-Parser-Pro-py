import pytest
from core.file_processor import FileProcessor
from core.nlp_engine import NlpEngine
from core.ml_models import MLModels
import os
from pathlib import Path

@pytest.fixture
def file_processor():
    return FileProcessor()

@pytest.fixture
def nlp_engine():
    return NlpEngine()

@pytest.fixture
def ml_models():
    return MLModels()

def test_pdf_processing(file_processor):
    test_file = Path(__file__).parent / "sample_resumes" / "sample.pdf"
    if not test_file.exists():
        pytest.skip("Sample PDF file not found")
    text = file_processor.extract_text(str(test_file))
    assert len(text) > 0
    assert "experience" in text.lower() or "education" in text.lower()

def test_docx_processing(file_processor):
    test_file = Path(__file__).parent / "sample_resumes" / "sample.docx"
    if not test_file.exists():
        pytest.skip("Sample DOCX file not found")
    text = file_processor.extract_text(str(test_file))
    assert len(text) > 0
    assert "experience" in text.lower() or "education" in text.lower()

def test_name_extraction(nlp_engine):
    test_text = "John Doe\nSenior Software Engineer\njohn.doe@example.com"
    entities = nlp_engine.extract_entities(test_text)
    assert entities["name"] == "John Doe"

def test_name_extraction_with_title(nlp_engine):
    test_text = "Dr. Jane Smith\nData Scientist\njane.smith@example.com"
    entities = nlp_engine.extract_entities(test_text)
    assert entities["name"] == "Jane Smith"

def test_name_extraction_labeled(nlp_engine):
    test_text = "Name: Robert Johnson\nEmail: robert.j@example.com"
    entities = nlp_engine.extract_entities(test_text)
    assert entities["name"] == "Robert Johnson"

def test_contact_extraction(nlp_engine):
    test_text = "Contact: john.doe@example.com, (123) 456-7890"
    entities = nlp_engine.extract_entities(test_text)
    assert entities["contact"]["email"] == "john.doe@example.com"
    assert "123" in entities["contact"]["phone"]

def test_contact_extraction_international_phone(nlp_engine):
    test_text = "Phone: +1-555-123-4567\nEmail: test@example.com"
    entities = nlp_engine.extract_entities(test_text)
    assert entities["contact"]["phone"] is not None
    assert "555" in entities["contact"]["phone"] or "123" in entities["contact"]["phone"]

def test_contact_extraction_linkedin(nlp_engine):
    test_text = "LinkedIn: https://linkedin.com/in/johndoe\nGitHub: github.com/johndoe"
    entities = nlp_engine.extract_entities(test_text)
    assert entities["contact"]["linkedin"] is not None
    assert "johndoe" in entities["contact"]["linkedin"]
    assert entities["contact"]["github"] is not None
    assert "johndoe" in entities["contact"]["github"]

def test_education_extraction_bachelors(nlp_engine):
    test_text = """
    Education
    Bachelor of Science in Computer Science
    Stanford University
    GPA: 3.8/4.0
    2018 - 2022
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["education"]) > 0

def test_education_extraction_masters(nlp_engine):
    test_text = """
    Education:
    M.S. in Machine Learning
    MIT - Massachusetts Institute of Technology
    September 2020 - May 2022
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["education"]) > 0

def test_education_extraction_mba(nlp_engine):
    test_text = """
    MBA, Harvard Business School, 2019
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["education"]) > 0

def test_experience_extraction(nlp_engine):
    test_text = """
    Experience
    
    Google Inc.
    Senior Software Engineer
    January 2020 - Present
    - Developed machine learning models
    - Led a team of 5 engineers
    
    Amazon
    Software Developer
    June 2018 - December 2019
    - Built RESTful APIs
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["experience"]) > 0

def test_experience_extraction_with_dates(nlp_engine):
    test_text = """
    Work History:
    Product Manager at Microsoft
    Mar 2021 - Present
    Managed product roadmap for Azure services
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["experience"]) > 0

def test_skills_extraction(nlp_engine):
    test_text = """
    Skills:
    Python, JavaScript, React, Node.js, AWS, Docker, Kubernetes
    Machine Learning, Deep Learning, TensorFlow, PyTorch
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["skills"]) > 0
    # Check for some expected skills
    skills_lower = [s.lower() for s in entities["skills"]]
    assert any("python" in s for s in skills_lower)

def test_skills_extraction_technical(nlp_engine):
    test_text = """
    Technical Skills
    • Programming Languages: Python, Java, C++, JavaScript
    • Frameworks: Django, React, Angular
    • Databases: PostgreSQL, MongoDB, Redis
    • Cloud: AWS, GCP, Azure
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["skills"]) > 0

def test_certifications_extraction(nlp_engine):
    test_text = """
    Certifications:
    - AWS Certified Solutions Architect - Professional
    - Google Cloud Professional Data Engineer
    - PMP - Project Management Professional
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["certifications"]) > 0

def test_projects_extraction(nlp_engine):
    test_text = """
    Projects
    
    E-commerce Platform - React, Node.js, MongoDB
    Built a full-stack e-commerce application with payment integration
    
    Machine Learning Pipeline - Python, TensorFlow
    Developed an automated ML pipeline for image classification
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["projects"]) > 0

def test_summary_extraction(nlp_engine):
    test_text = """
    Professional Summary
    
    Experienced software engineer with 8+ years of experience in building scalable 
    applications. Passionate about machine learning and cloud technologies.
    
    Skills:
    Python, AWS, Kubernetes
    """
    entities = nlp_engine.extract_entities(test_text)
    assert entities["summary"] is not None
    assert "experienced" in entities["summary"].lower() or "software" in entities["summary"].lower()

def test_languages_extraction(nlp_engine):
    test_text = """
    Languages:
    - English (Native)
    - Spanish (Fluent)
    - French (Intermediate)
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["languages"]) > 0
    languages_list = [l["language"].lower() for l in entities["languages"]]
    assert "english" in languages_list or "spanish" in languages_list

def test_achievements_extraction(nlp_engine):
    test_text = """
    Achievements:
    - Awarded Employee of the Year 2022
    - Promoted to Senior Engineer in 18 months
    - Received patent for innovative caching algorithm
    """
    entities = nlp_engine.extract_entities(test_text)
    assert len(entities["achievements"]) > 0

def test_full_resume_parsing(nlp_engine):
    """Test parsing a complete resume"""
    test_text = """
    John Smith
    Senior Software Engineer
    
    Contact: john.smith@email.com | (555) 123-4567
    LinkedIn: linkedin.com/in/johnsmith | GitHub: github.com/johnsmith
    Location: San Francisco, CA
    
    Professional Summary
    Results-driven software engineer with 7+ years of experience building 
    high-performance applications. Expert in Python, cloud technologies, and 
    machine learning.
    
    Experience
    
    Tech Corp Inc.
    Senior Software Engineer
    January 2020 - Present
    • Led development of microservices architecture
    • Improved system performance by 40%
    • Mentored junior developers
    
    StartupXYZ
    Software Developer
    June 2017 - December 2019
    • Built RESTful APIs using Python and Django
    • Implemented CI/CD pipelines
    
    Education
    
    Stanford University
    Bachelor of Science in Computer Science
    GPA: 3.9/4.0
    2013 - 2017
    
    Skills
    Python, Java, JavaScript, React, Node.js, Django, Flask
    AWS, GCP, Docker, Kubernetes, Terraform
    PostgreSQL, MongoDB, Redis
    Machine Learning, TensorFlow, PyTorch
    
    Certifications
    - AWS Certified Solutions Architect - Professional
    - Google Cloud Professional Cloud Architect
    
    Languages
    - English (Native)
    - Mandarin (Conversational)
    
    Achievements
    - Awarded "Innovation Award" for ML project
    - Published paper in IEEE conference
    """
    
    entities = nlp_engine.extract_entities(test_text)
    
    # Verify all sections are extracted
    assert entities["name"] is not None
    assert entities["contact"]["email"] is not None
    assert entities["summary"] is not None
    assert len(entities["experience"]) >= 1
    assert len(entities["education"]) >= 1
    assert len(entities["skills"]) >= 1
    assert len(entities["certifications"]) >= 1
    assert len(entities["languages"]) >= 1

def test_skill_normalization(ml_models):
    skills = ["python programming", "ml", "data analysis"]
    normalized = ml_models.normalize_skills(skills)
    assert "Python" in normalized
    assert "Machine Learning" in normalized
    assert "Data Analysis" in normalized

def test_compatibility_scoring(ml_models):
    resume_data = {
        "skills": ["Python", "Machine Learning"],
        "experience": [{"company": "Google", "position": "Data Scientist"}],
        "education": [{"degree": "PhD", "institution": "Stanford University"}]
    }
    job_desc = {
        "title": "Data Scientist",
        "description": "Looking for a Python expert with ML experience",
        "requirements": ["Python", "Machine Learning", "Data Analysis"]
    }
    score = ml_models.calculate_compatibility(resume_data, job_desc)
    assert score["overall_score"] > 0.5
    assert score["skill_match"] > 0.5