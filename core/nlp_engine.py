import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
import os
from datetime import datetime

class NlpEngine:
    def __init__(self):
        # Try loading models in order of preference (lg > md > sm)
        model_name = os.environ.get("SPACY_MODEL", "en_core_web_sm")
        for model in [model_name, "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
            try:
                self.nlp = spacy.load(model)
                self.model_name = model
                break
            except OSError:
                continue
        else:
            raise RuntimeError("No spaCy English model found. Install with: python -m spacy download en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self._initialize_matchers()
        self._load_patterns()
        self._load_common_words()
        
        # Add custom pipeline components
        Span.set_extension("score", default=1.0, force=True)
    
    def _load_common_words(self):
        """Load common words to skip in name extraction"""
        self.skip_words = {
            'resume', 'cv', 'curriculum', 'vitae', 'profile', 'summary', 'objective',
            'experience', 'education', 'skills', 'contact', 'about', 'me', 'professional',
            'personal', 'information', 'details', 'material', 'receipt', 'document',
            'page', 'linkedin', 'github', 'email', 'phone', 'address', 'career',
            'work', 'employment', 'history', 'qualifications', 'certifications',
            'references', 'available', 'upon', 'request', 'the', 'and', 'for', 'with',
            'top', 'languages', 'analytical', 'supervisory', 'shipping', 'logistics',
            'production', 'specialist', 'focused', 'efficiency', 'team', 'development',
            'operational', 'excellence', 'ontario', 'canada', 'punjabi', 'hindi', 'english',
            'native', 'bilingual', 'working', 'certified', 'international', 'freight',
            'forwarder', 'dedicated', 'diploma', 'supply', 'chain', 'management',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december', 'present', 'current',
            'software', 'engineer', 'developer', 'manager', 'director', 'analyst',
            'consultant', 'senior', 'junior', 'lead', 'head', 'chief', 'vice', 'president'
        }
        
        self.job_titles = {
            'software engineer', 'senior software engineer', 'data scientist', 
            'machine learning engineer', 'product manager', 'project manager',
            'business analyst', 'data analyst', 'frontend developer', 'backend developer',
            'full stack developer', 'devops engineer', 'cloud architect', 'solutions architect',
            'ux designer', 'ui designer', 'graphic designer', 'marketing manager',
            'sales manager', 'account manager', 'hr manager', 'finance manager',
            'operations manager', 'supply chain manager', 'logistics manager',
            'quality assurance', 'qa engineer', 'test engineer', 'security engineer',
            'network engineer', 'systems administrator', 'database administrator',
            'technical writer', 'content writer', 'copywriter', 'research scientist',
            'research engineer', 'postdoctoral researcher', 'professor', 'lecturer',
            'teaching assistant', 'intern', 'trainee', 'associate', 'specialist',
            'coordinator', 'executive', 'officer', 'administrator', 'supervisor',
            'team lead', 'tech lead', 'engineering manager', 'cto', 'ceo', 'cfo', 'coo',
            'founder', 'co-founder', 'partner', 'principal', 'consultant'
        }
        
        self.degree_keywords = {
            'bachelor', 'master', 'phd', 'doctorate', 'associate', 'diploma',
            'certificate', 'degree', 'bsc', 'msc', 'ba', 'ma', 'mba', 'bba',
            'btech', 'mtech', 'be', 'me', 'bcom', 'mcom', 'bca', 'mca', 'llb', 'llm',
            'md', 'mbbs', 'bds', 'pharm', 'nursing'
        }
        
        self.section_headers = {
            'experience', 'education', 'skills', 'projects', 'certifications',
            'achievements', 'awards', 'publications', 'languages', 'interests',
            'hobbies', 'references', 'summary', 'objective', 'profile', 'about',
            'work history', 'employment history', 'professional experience',
            'academic background', 'technical skills', 'soft skills', 'core competencies'
        }
        
    def _initialize_matchers(self):
        """Initialize various matchers for entity extraction"""
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Education patterns - expanded for more degree types
        education_patterns = [
            # Bachelor's degrees
            [{"LOWER": {"IN": ["bsc", "b.sc", "bs", "b.s", "b.sc."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["btech", "b.tech", "b.tech."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["be", "b.e", "b.e."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["ba", "b.a", "b.a."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["bba", "b.b.a", "bba."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["bcom", "b.com", "b.com."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["bca", "b.c.a", "bca."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": "bachelor"}, {"LOWER": "of"}, {"IS_ALPHA": True}],
            [{"LOWER": "bachelor"}, {"LOWER": "'s"}, {"IS_ALPHA": True, "OP": "?"}],
            # Master's degrees
            [{"LOWER": {"IN": ["msc", "m.sc", "ms", "m.s", "m.sc."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["mtech", "m.tech", "m.tech."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["me", "m.e", "m.e."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["ma", "m.a", "m.a."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["mba", "m.b.a", "mba."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["mcom", "m.com", "m.com."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["mca", "m.c.a", "mca."]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": "master"}, {"LOWER": "of"}, {"IS_ALPHA": True}],
            [{"LOWER": "master"}, {"LOWER": "'s"}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": "ms"}, {"LOWER": "in"}, {"IS_ALPHA": True}],
            # Doctoral degrees
            [{"LOWER": {"IN": ["phd", "ph.d", "ph.d.", "d.phil"]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": "doctor"}, {"LOWER": "of"}, {"IS_ALPHA": True}],
            [{"LOWER": "doctorate"}, {"IS_ALPHA": True, "OP": "?"}],
            # Associate and diploma
            [{"LOWER": "associate"}, {"LOWER": "of"}, {"IS_ALPHA": True}],
            [{"LOWER": "associate"}, {"LOWER": "'s"}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": "diploma"}, {"LOWER": "in"}, {"IS_ALPHA": True}],
            [{"LOWER": "advanced"}, {"LOWER": "diploma"}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": "postgraduate"}, {"LOWER": "diploma"}, {"IS_ALPHA": True, "OP": "?"}],
            # Law degrees
            [{"LOWER": {"IN": ["llb", "ll.b", "ll.b.", "jd", "j.d"]}}, {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["llm", "ll.m", "ll.m."]}}, {"IS_ALPHA": True, "OP": "?"}],
            # Medical degrees
            [{"LOWER": {"IN": ["md", "m.d", "m.d.", "mbbs", "m.b.b.s"]}}, {"IS_ALPHA": True, "OP": "?"}],
        ]
        self.matcher.add("EDUCATION", education_patterns)
        
        # Experience patterns - expanded
        experience_patterns = [
            [{"LOWER": {"IN": ["worked", "work", "working"]}}, {"LOWER": "as"}, {"POS": "NOUN", "OP": "+"}],
            [{"LOWER": {"IN": ["worked", "work", "working"]}}, {"LOWER": "at"}, {"POS": "PROPN", "OP": "+"}],
            [{"LOWER": {"IN": ["worked", "work", "working"]}}, {"LOWER": "for"}, {"POS": "PROPN", "OP": "+"}],
            [{"LOWER": {"IN": ["joined", "join"]}}, {"LOWER": "as"}, {"POS": "NOUN", "OP": "+"}],
            [{"LOWER": {"IN": ["employed", "hired"]}}, {"LOWER": "as"}, {"POS": "NOUN", "OP": "+"}],
            [{"LOWER": {"IN": ["employed", "hired"]}}, {"LOWER": "by"}, {"POS": "PROPN", "OP": "+"}],
        ]
        self.matcher.add("EXPERIENCE", experience_patterns)
        
        # Date patterns for experience/education - comprehensive for all formats
        date_patterns = [
            # Month Year - Month Year or Present (standard US)
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "-"}, {"LOWER": "present"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "–"}, {"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "–"}, {"LOWER": "present"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "—"}, {"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "—"}, {"LOWER": "present"}],
            # MM/YYYY - MM/YYYY (ATS-friendly)
            [{"SHAPE": "dd/dddd"}, {"ORTH": "-"}, {"SHAPE": "dd/dddd"}],
            [{"SHAPE": "dd/dddd"}, {"ORTH": "-"}, {"LOWER": "present"}],
            [{"SHAPE": "dd/dddd"}, {"ORTH": "–"}, {"SHAPE": "dd/dddd"}],
            [{"SHAPE": "dd/dddd"}, {"ORTH": "–"}, {"LOWER": "present"}],
            # Year only: 2020 - 2023
            [{"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "dddd"}, {"ORTH": "-"}, {"LOWER": "present"}],
            [{"SHAPE": "dddd"}, {"ORTH": "–"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "dddd"}, {"ORTH": "–"}, {"LOWER": "present"}],
            # Abbreviated month: Jan 2020 - Dec 2021
            [{"SHAPE": "Xxx"}, {"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "Xxx"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "Xxx"}, {"SHAPE": "dddd"}, {"ORTH": "-"}, {"LOWER": "present"}],
            # "to" instead of dash: Jan 2020 to Dec 2021
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"LOWER": "to"}, {"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"LOWER": "to"}, {"LOWER": "present"}],
            # Current/Now variations
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "-"}, {"LOWER": "current"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "-"}, {"LOWER": "now"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "–"}, {"LOWER": "current"}],
        ]
        self.matcher.add("DATE_RANGE", date_patterns)
        
    def _load_patterns(self):
        """Load patterns from external files"""
        data_dir = Path(__file__).parent.parent / "data"
        
        # Load skills
        skills_file = data_dir / "skills.json"
        with open(skills_file, 'r') as f:
            skills_data = json.load(f)
        self.skill_patterns = list(self.nlp.pipe(skills_data["skills"]))
        self.phrase_matcher.add("SKILLS", self.skill_patterns)
        
        # Load companies
        companies_file = data_dir / "companies.json"
        with open(companies_file, 'r') as f:
            companies_data = json.load(f)
        self.company_patterns = list(self.nlp.pipe(companies_data["companies"]))
        self.phrase_matcher.add("COMPANIES", self.company_patterns)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from resume text"""
        doc = self.nlp(text)
        
        # Run matchers
        matches = self.matcher(doc)
        phrase_matches = self.phrase_matcher(doc)
        
        entities = {
            "name": self._extract_name(doc),
            "contact": self._extract_contact_info(doc),
            "summary": self._extract_summary(doc),
            "education": self._extract_education(doc, matches),
            "experience": self._extract_experience(doc, matches, phrase_matches),
            "skills": self._extract_skills(doc, phrase_matches),
            "certifications": self._extract_certifications(doc),
            "projects": self._extract_projects(doc),
            "languages": self._extract_languages(doc),
            "achievements": self._extract_achievements(doc)
        }
        
        return entities
    
    def _extract_name(self, doc) -> Optional[str]:
        """Extract candidate name from document - handles multiple resume formats
        
        Supports various resume layouts:
        - Traditional: Name at top center/left
        - Modern: Name in header with contact info
        - Creative: Name with styling/formatting
        - Academic CV: Name followed by credentials
        - ATS-friendly: Simple name at top
        - Two-column: Name in sidebar or main area
        """
        # Common titles/prefixes to strip
        title_prefixes = {'mr', 'mrs', 'ms', 'dr', 'prof', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sir', 'madam'}
        
        def is_valid_name(text: str) -> bool:
            """Check if text looks like a valid person name"""
            words = text.strip().split()
            if len(words) < 2 or len(words) > 5:
                return False
            # Skip if any word is a common skip word
            for word in words:
                clean_word = word.lower().strip('.,')
                if clean_word in self.skip_words:
                    return False
                # Skip if it looks like a job title
                if clean_word in {'engineer', 'developer', 'manager', 'analyst', 'consultant', 
                                  'director', 'specialist', 'coordinator', 'lead', 'senior', 
                                  'junior', 'intern', 'associate', 'resume', 'curriculum', 'vitae'}:
                    return False
            # All words should be primarily alphabetic and title case or all caps
            for word in words:
                clean_word = word.replace('.', '').replace(',', '').replace("'", "").replace('-', '')
                if not clean_word.isalpha():
                    return False
                # Should be title case or all caps (like "JOHN DOE")
                if not (word[0].isupper() or word.isupper()):
                    return False
            # Name shouldn't be too long (each word) or too short
            if any(len(w.replace('.', '')) > 15 or len(w.replace('.', '')) < 2 for w in words):
                return False
            return True
        
        def clean_name(name: str) -> str:
            """Clean and standardize name format"""
            words = name.strip().split()
            # Remove title prefixes
            while words and words[0].lower().strip('.') in title_prefixes:
                words = words[1:]
            # Remove common suffixes
            suffixes = {'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'phd', 'md', 'esq', 'mba', 'cpa'}
            while words and words[-1].lower().strip('.') in suffixes:
                words = words[:-1]
            if len(words) >= 2:
                return ' '.join(word.title() if not word.isupper() else word.title() for word in words)
            return name.title()
        
        text = doc.text
        
        # Strategy 1: Look for explicit name labels (common in forms and some templates)
        name_label_patterns = [
            r'(?:name|full\s*name|candidate\s*name|applicant\s*name)\s*[:\-]\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\n',  # Name at start of doc followed by newline
            # ALL CAPS name pattern (common in professional resumes)
            r'^([A-Z]{2,}(?:\s+[A-Z]{2,})+)\s*\n',
        ]
        for pattern in name_label_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                if is_valid_name(potential_name):
                    return clean_name(potential_name)
        
        # Strategy 2: Look for "FirstName LastName" pattern followed by job title keywords
        job_title_keywords = ['specialist', 'manager', 'engineer', 'developer', 'analyst', 
                              'consultant', 'director', 'coordinator', 'associate', 'lead',
                              'supervisor', 'executive', 'administrator', 'architect',
                              'designer', 'scientist', 'researcher', 'accountant', 'officer',
                              'intern', 'trainee', 'founder', 'partner', 'professional']
        for keyword in job_title_keywords:
            # Match "FirstName LastName" before job title (with optional middle words)
            pattern = rf'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:[A-Za-z&|\s]{{0,30}})?{keyword}'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                # Only take first 2-3 words (avoid capturing "John Doe Senior Software" as name)
                words = potential_name.split()[:3]
                potential_name = ' '.join(words)
                if is_valid_name(potential_name):
                    return clean_name(potential_name)
        
        # Strategy 3: Look for name pattern after LinkedIn URL (common in LinkedIn PDFs)
        linkedin_name_pattern = r'\(LinkedIn\)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        match = re.search(linkedin_name_pattern, text)
        if match:
            potential_name = match.group(1).strip()
            if is_valid_name(potential_name):
                return clean_name(potential_name)
        
        # Strategy 4: Look for PERSON entities from spaCy NER (first valid one)
        person_candidates = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if is_valid_name(ent.text):
                    person_candidates.append((ent.text, ent.start_char))
        
        # Prefer the earliest PERSON entity (usually at top of resume)
        if person_candidates:
            person_candidates.sort(key=lambda x: x[1])
            return clean_name(person_candidates[0][0])
        
        # Strategy 5: Look at first few lines for standalone name patterns
        lines = text.split('\n')
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if not line or len(line) < 4:
                continue
            # Skip lines with URLs, emails, or phone numbers
            if any(skip in line.lower() for skip in ['@', 'http', 'www.', '.com', '.org', '.edu', 'phone', 'tel:', 'email:']):
                continue
            # Skip if line looks like a section header
            if any(header in line.lower() for header in self.section_headers):
                continue
            # Skip lines with numbers (likely phone or address)
            if re.search(r'\d{3,}', line):
                continue
            
            words = line.split()
            if 2 <= len(words) <= 4:
                clean_words = [w for w in words if w.lower().strip('.') not in title_prefixes]
                if len(clean_words) >= 2:
                    potential_name = ' '.join(clean_words)
                    if is_valid_name(potential_name):
                        return clean_name(potential_name)
        
        # Strategy 6: Look for email-based name extraction
        email_pattern = r'([a-zA-Z]+)\.([a-zA-Z]+)@'
        match = re.search(email_pattern, text)
        if match:
            first_name = match.group(1).title()
            last_name = match.group(2).title()
            potential_name = f"{first_name} {last_name}"
            # Only use if we couldn't find name otherwise and it looks valid
            if is_valid_name(potential_name):
                return potential_name
        
        return None
    
    def _extract_contact_info(self, doc) -> Dict[str, Optional[str]]:
        """Extract contact information (email, phone, website, linkedin, location)"""
        text = doc.text
        contact = {
            "email": None, 
            "phone": None, 
            "website": None, 
            "linkedin": None,
            "github": None,
            "location": None
        }
        
        # ===================
        # Extract Email
        # ===================
        email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
        emails = re.findall(email_regex, text)
        if emails:
            # Filter out common non-personal emails
            personal_emails = [e for e in emails if not any(
                skip in e.lower() for skip in ['noreply', 'no-reply', 'example.com', 'test.com']
            )]
            contact["email"] = personal_emails[0] if personal_emails else emails[0]
        
        # ===================
        # Extract Phone Numbers
        # ===================
        phone_patterns = [
            # International format: +1 (123) 456-7890 or +91-9876543210
            r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            # US/Canada format: (123) 456-7890 or 123-456-7890 or 123.456.7890
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            # With country code: 1-800-123-4567
            r'\d{1}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            # Indian format: 98765 43210 or 9876543210
            r'\b\d{5}[-.\s]?\d{5}\b',
            # General 10+ digit number
            r'\b\d{10,14}\b',
        ]
        
        phone_candidates = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phone_candidates.extend(matches)
        
        if phone_candidates:
            # Clean and deduplicate phone numbers
            cleaned_phones = []
            for phone in phone_candidates:
                # Clean the phone number
                cleaned = re.sub(r'[^\d+]', '', phone)
                # Must be at least 10 digits
                if len(cleaned.replace('+', '')) >= 10:
                    cleaned_phones.append(phone.strip())
            
            if cleaned_phones:
                # Prefer international format, otherwise take first
                intl_phones = [p for p in cleaned_phones if p.startswith('+')]
                contact["phone"] = intl_phones[0] if intl_phones else cleaned_phones[0]
        
        # ===================
        # Extract LinkedIn
        # ===================
        # LinkedIn usernames can contain letters, numbers, hyphens (max ~100 chars)
        # LinkedIn URLs format: linkedin.com/in/username or linkedin.com/in/firstname-lastname-uuid
        # The UUID part after the name is important and should be preserved
        linkedin_patterns = [
            # Full URL with optional UUID suffix (e.g., bhavjot-singh-1a2b3c4d)
            r'https?://(?:www\.)?linkedin\.com/in/([a-zA-Z0-9][a-zA-Z0-9_-]{2,100})',
            r'(?:www\.)?linkedin\.com/in/([a-zA-Z0-9][a-zA-Z0-9_-]{2,100})',
        ]
        for pattern in linkedin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Clean up the username - handle PDF extraction artifacts
                username = match.group(1)
                
                # Remove common PDF artifacts - text that gets concatenated after URL
                # Stop at common resume section headers that get merged
                stop_words = ['Contact', 'About', 'Experience', 'Education', 'Skills', 
                              'Top', 'Summary', 'Page', 'LinkedIn', 'Member']
                for stop_word in stop_words:
                    if stop_word in username:
                        idx = username.find(stop_word)
                        if idx > 5:  # Keep at least 5 chars
                            username = username[:idx]
                            break
                
                # LinkedIn usernames are lowercase in the URL
                clean_username = username.lower()
                
                # Remove any trailing hyphens or special chars
                clean_username = clean_username.rstrip('-_/')
                
                # If username ends with numbers after a hyphen, it's likely the UUID - keep it
                # Format: firstname-lastname-uuid123 or just username-uuid123
                
                # LinkedIn usernames are typically 5-100 chars
                # Validate the cleaned username
                if 3 <= len(clean_username) <= 100:
                    # Check it looks like a valid LinkedIn URL (has at least some lowercase letters)
                    if re.match(r'^[a-z0-9][a-z0-9_-]*[a-z0-9]$', clean_username) or len(clean_username) <= 5:
                        contact["linkedin"] = f"https://linkedin.com/in/{clean_username}"
                        break
        
        # ===================
        # Extract GitHub
        # ===================
        github_patterns = [
            r'github\.com/([a-zA-Z0-9_-]+[a-zA-Z0-9])',  # Must end with alphanumeric
            r'www\.github\.com/([a-zA-Z0-9_-]+[a-zA-Z0-9])',
            r'https?://(?:www\.)?github\.com/([a-zA-Z0-9_-]+[a-zA-Z0-9])',
        ]
        for pattern in github_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                username = match.group(1).rstrip('/')
                # Skip common non-user paths
                if username.lower() not in ['login', 'signup', 'explore', 'settings', 'notifications']:
                    contact["github"] = f"https://github.com/{username}"
                    break
        
        # ===================
        # Extract Personal Website/Portfolio
        # ===================
        website_patterns = [
            # Personal domains
            r'https?://(?:www\.)?([a-zA-Z0-9-]+\.(?:com|io|dev|me|co|org|net|tech|app|xyz|portfolio|site)(?:/[^\s]*)?)',
            # Portfolio sites
            r'https?://([a-zA-Z0-9-]+\.(?:github\.io|netlify\.app|vercel\.app|herokuapp\.com|pages\.dev)(?:/[^\s]*)?)',
            # Simple www pattern
            r'www\.([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(?:/[^\s]*)?',
        ]
        
        skip_sites = ['linkedin.com', 'github.com', 'twitter.com', 'facebook.com', 
                      'instagram.com', 'youtube.com', 'gmail.com', 'yahoo.com', 
                      'hotmail.com', 'outlook.com', 'google.com']
        
        for pattern in website_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                site = match if isinstance(match, str) else match[0] if match else None
                if site and not any(skip in site.lower() for skip in skip_sites):
                    # Ensure it has protocol
                    if not site.startswith('http'):
                        site = f"https://{site}"
                    contact["website"] = site
                    break
            if contact["website"]:
                break
        
        # ===================
        # Extract Location
        # ===================
        # Skip words that shouldn't be part of location
        location_skip_words = {'excellence', 'professional', 'experience', 'skills', 
                               'summary', 'objective', 'focused', 'specialist'}
        
        location_patterns = [
            # City, Province/State, Country - be more specific
            r'\b(Etobicoke|Toronto|Vancouver|Calgary|Montreal|Ottawa|Mississauga|Brampton|Hamilton|Kitchener|London|Victoria|Edmonton|Winnipeg|Halifax|Saskatoon|Regina),\s*(Ontario|Quebec|British Columbia|Alberta|Manitoba|Saskatchewan|Nova Scotia),\s*(Canada)\b',
            # US Cities
            r'\b(New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Seattle|Denver|Boston|Portland),\s*(NY|CA|TX|AZ|PA|WA|CO|MA|OR|[A-Z][a-z]+),?\s*(USA|United States)?\b',
            # Indian Cities
            r'\b(Delhi|Mumbai|Bangalore|Bengaluru|Chennai|Hyderabad|Kolkata|Pune|Ahmedabad|Jaipur|Lucknow|Chandigarh),\s*(India|[A-Z][a-z]+)?\b',
            # City, Province/State (without country)
            r'\b(Etobicoke|Toronto|Vancouver|Calgary|Montreal|Ottawa|Mississauga|Brampton),\s*(Ontario|Quebec|British Columbia|Alberta)\b',
            # Generic: City, State/Province, Country
            r'\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)?),\s*(Ontario|Quebec|British Columbia|Alberta|California|New York|Texas),\s*(Canada|USA)\b',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(0)
                # Verify no skip words in the match
                if not any(skip in location.lower() for skip in location_skip_words):
                    contact["location"] = location
                    break
        
        # Fallback: Look for specific Canadian/US location patterns
        if not contact["location"]:
            # Look for "City, Province, Canada" pattern more broadly
            broad_pattern = r'\b([A-Z][a-z]+),\s*(Ontario|British Columbia|Alberta|Quebec),\s*Canada\b'
            match = re.search(broad_pattern, text)
            if match:
                contact["location"] = match.group(0)
        
        # Last resort: spaCy GPE entities (but filter out common false positives)
        if not contact["location"]:
            gpe_skip = {'linkedin', 'github', 'india', 'excellence', 'canada', 'usa'}
            locations = [ent.text for ent in doc.ents 
                        if ent.label_ == "GPE" 
                        and ent.text.lower() not in gpe_skip
                        and len(ent.text) > 3]
            if locations:
                # Filter to keep only likely location names
                valid_locations = [loc for loc in locations 
                                   if loc[0].isupper() and loc.replace(' ', '').isalpha()]
                if valid_locations:
                    contact["location"] = ", ".join(valid_locations[:2])
        
        return contact
    
    def _extract_education(self, doc, matches) -> List[Dict[str, Any]]:
        """Extract comprehensive education information"""
        education = []
        text = doc.text
        
        # Common degree patterns with their full names
        degree_mapping = {
            'phd': 'Doctor of Philosophy (Ph.D.)',
            'ph.d': 'Doctor of Philosophy (Ph.D.)',
            'ph.d.': 'Doctor of Philosophy (Ph.D.)',
            'doctorate': 'Doctorate',
            'mba': 'Master of Business Administration (MBA)',
            'm.b.a': 'Master of Business Administration (MBA)',
            'msc': 'Master of Science (M.Sc.)',
            'm.sc': 'Master of Science (M.Sc.)',
            'ms': 'Master of Science (M.S.)',
            'm.s': 'Master of Science (M.S.)',
            'ma': 'Master of Arts (M.A.)',
            'm.a': 'Master of Arts (M.A.)',
            'mtech': 'Master of Technology (M.Tech)',
            'm.tech': 'Master of Technology (M.Tech)',
            'me': 'Master of Engineering (M.E.)',
            'm.e': 'Master of Engineering (M.E.)',
            'mca': 'Master of Computer Applications (MCA)',
            'mcom': 'Master of Commerce (M.Com)',
            'llm': 'Master of Laws (LL.M)',
            'bsc': 'Bachelor of Science (B.Sc.)',
            'b.sc': 'Bachelor of Science (B.Sc.)',
            'bs': 'Bachelor of Science (B.S.)',
            'b.s': 'Bachelor of Science (B.S.)',
            'ba': 'Bachelor of Arts (B.A.)',
            'b.a': 'Bachelor of Arts (B.A.)',
            'btech': 'Bachelor of Technology (B.Tech)',
            'b.tech': 'Bachelor of Technology (B.Tech)',
            'be': 'Bachelor of Engineering (B.E.)',
            'b.e': 'Bachelor of Engineering (B.E.)',
            'bba': 'Bachelor of Business Administration (BBA)',
            'bca': 'Bachelor of Computer Applications (BCA)',
            'bcom': 'Bachelor of Commerce (B.Com)',
            'llb': 'Bachelor of Laws (LL.B)',
            'mbbs': 'Bachelor of Medicine, Bachelor of Surgery (MBBS)',
            'md': 'Doctor of Medicine (M.D.)',
        }
        
        # Institution keywords
        institution_keywords = ['university', 'college', 'institute', 'school', 'academy', 
                                'polytechnic', 'iit', 'nit', 'iiit', 'bits', 'mit', 'stanford',
                                'harvard', 'oxford', 'cambridge']
        
        # Extract education section
        education_section = self._extract_section(text, ['education', 'academic background', 
                                                          'educational qualifications', 'qualifications',
                                                          'academic qualifications'])
        
        if education_section:
            # Parse education entries from the section
            entries = self._parse_education_section(education_section, degree_mapping, institution_keywords)
            education.extend(entries)
        
        # Fallback: Extract using NER for institutions (only if no education found yet)
        if not education:
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    ent_lower = ent.text.lower()
                    # Must contain institution keyword and look like a real institution
                    if any(kw in ent_lower for kw in institution_keywords):
                        # Skip false positives
                        skip_patterns = ['management', 'asset', 'monitor', 'project', 'resource', 
                                        'planning', 'control', 'quality', 'leadership']
                        if any(skip in ent_lower for skip in skip_patterns):
                            continue
                        # Must be at least 3 words for a real institution name or contain "university"/"college"
                        words = ent.text.split()
                        if len(words) >= 2 or any(kw in ent_lower for kw in ['university', 'college']):
                            education.append({
                                "institution": ent.text,
                                "degree": None,
                                "field_of_study": None,
                                "graduation_date": None,
                                "gpa": None
                            })
        
        # Extract using matcher patterns - only if we have NO education entries yet
        if not education:
            for match_id, start, end in matches:
                if self.nlp.vocab.strings[match_id] == "EDUCATION":
                    span = doc[start:end]
                    degree_text = span.text.lower().replace('.', '').replace(' ', '')
                    # Normalize the degree
                    for abbr, full_name in degree_mapping.items():
                        if degree_text.startswith(abbr.replace('.', '')):
                            # Check if we already have this degree
                            existing = any(full_name in str(e.get('degree', '')) for e in education)
                            if not existing:
                                education.append({
                                    "institution": None,
                                    "degree": full_name,
                                    "field_of_study": None,
                                    "graduation_date": None,
                                    "gpa": None
                                })
                            break
        
        # Clean and deduplicate
        unique_education = []
        seen_institutions = set()
        seen_degrees = set()
        
        for edu in education:
            institution = (edu.get('institution') or '').strip()
            degree = (edu.get('degree') or '').strip()
            field = (edu.get('field_of_study') or '').strip()
            
            # Clean up institution name - remove merged dates/pipes
            if institution:
                # Remove date patterns from institution name (e.g., "Fleming College | 2022-2023")
                institution = re.sub(r'\s*\|\s*\d{4}.*$', '', institution).strip()
                institution = re.sub(r'\s*\(\s*\d{4}.*$', '', institution).strip()
                # Remove newlines and normalize spacing
                institution = ' '.join(institution.split())
                
                # Check if field of study is merged with institution (e.g., "Entrepreneurship/Studies Canadore College")
                # Look for institution keyword and extract just that part
                for kw in institution_keywords:
                    kw_lower = kw.lower()
                    inst_lower = institution.lower()
                    if kw_lower in inst_lower:
                        # Find position of institution keyword
                        idx = inst_lower.find(kw_lower)
                        if idx > 0:
                            # The institution keyword is in the middle - likely field of study is prepended
                            # Extract text starting from a word before the keyword that looks like institution start
                            # Usually institution name starts 1-3 words before the keyword
                            words = institution.split()
                            kw_word_idx = None
                            for i, word in enumerate(words):
                                if kw_lower in word.lower():
                                    kw_word_idx = i
                                    break
                            if kw_word_idx is not None and kw_word_idx > 0:
                                # Take 1-2 words before the keyword as institution start (e.g., "Fleming" before "College")
                                start_idx = max(0, kw_word_idx - 2)
                                # But only if those words look like proper names (capitalized)
                                candidate_start = start_idx
                                for j in range(start_idx, kw_word_idx):
                                    if words[j][0].isupper() and not any(c in words[j].lower() for c in ['/', 'entrepreneurship', 'science', 'arts', 'studies', 'engineering', 'business']):
                                        candidate_start = j
                                        break
                                institution = ' '.join(words[candidate_start:])
                        break
                
                # Handle newline-separated field of study (e.g., "Entrepreneurship/Entrepreneurial Studies\nCanadore College")
                if '\n' in institution or len(institution.split()) > 8:
                    # Try to extract just the institution part
                    for kw in institution_keywords:
                        if kw in institution.lower():
                            # Find the part with the institution keyword
                            parts = re.split(r'[\n/]', institution)
                            for part in parts:
                                if kw in part.lower():
                                    institution = part.strip()
                                    break
                            break
                edu['institution'] = institution
            
            # Clean up field of study - remove trailing special chars and incomplete text
            if field:
                # Remove trailing · and incomplete parentheses
                field = re.sub(r'\s*·\s*\(?[A-Za-z]*$', '', field).strip()
                edu['field_of_study'] = field if field else None
            
            # Skip entries with suspicious institution names
            if institution:
                inst_lower = institution.lower()
                # Skip if contains action words (not institution names)
                skip_words = ['monitor', 'management:', 'asset', 'track', 'control', 
                             'provide', 'develop', 'ensure', 'implement', 'maintain',
                             'opportunities', 'hybrid', 'remote', 'seeking', 'looking']
                if any(sw in inst_lower for sw in skip_words):
                    continue
                # Skip if too short
                if len(institution) < 5:
                    continue
                # Skip if doesn't contain any institution keyword (must have university/college/institute etc)
                if not any(kw in inst_lower for kw in institution_keywords):
                    # Allow if it's a well-known institution abbreviation
                    known_abbrevs = ['iit', 'nit', 'iiit', 'bits', 'mit', 'stanford', 'harvard', 'oxford', 'cambridge']
                    if not any(abbr in inst_lower for abbr in known_abbrevs):
                        continue
                # Skip if starts with lowercase or common words
                if institution[0].islower() or institution.lower().startswith(('and ', 'or ', 'the ', 'a ')):
                    continue
            
            # Skip entries that are only degree with no institution (if we have better entries)
            if not institution and degree:
                # Check if we already have this degree with an institution
                degree_lower = degree.lower()
                if degree_lower in seen_degrees:
                    continue
                seen_degrees.add(degree_lower)
            
            # Track institutions to avoid duplicates
            if institution:
                inst_lower = institution.lower()
                if inst_lower in seen_institutions:
                    continue
                seen_institutions.add(inst_lower)
            
            unique_education.append(edu)
        
        return unique_education
    
    def _parse_education_section(self, section: str, degree_mapping: Dict[str, str], 
                                  institution_keywords: List[str]) -> List[Dict[str, Any]]:
        """Parse education entries from a section - handles multiple resume formats
        
        Supported formats:
        - LinkedIn PDF export
        - Traditional chronological (Institution -> Degree -> Dates)
        - ATS-friendly (Degree | Institution | Date)
        - Academic CV (detailed with GPA, honors, coursework)
        - Functional resumes (grouped by relevance)
        - European/Europass format
        - Single-line compact format
        """
        entries = []
        
        # LinkedIn format patterns
        # Date range pattern: (September 2022 - April 2023) or (2018 - 2021)
        date_range_pattern = r'\(?\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?\s*\d{4})\s*[-–]\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?\s*\d{4}|Present|Current)\s*\)?'
        single_date_pattern = r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}|\d{4})'
        gpa_pattern = r'(?:GPA|CGPA|Grade|Score)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?'
        
        # LinkedIn format: "Degree, Field of Study · (Date Range)"
        linkedin_degree_pattern = r'^([^,·\n]+),\s*([^·\n]+?)(?:\s*·\s*|\s+)\((.+?)\)\s*$'
        
        # Format 2: Traditional - "Degree in Field" or "Degree (Field)"
        traditional_degree_pattern = r'^([A-Za-z\.\s]+(?:Bachelor|Master|Doctor|PhD|Associate|Diploma|Certificate)[A-Za-z\.\s]*)(?:\s+in\s+|\s*[\(\[]\s*)([^\)\]\|]+)?(?:[\)\]])?'
        
        # Format 3: ATS-friendly - "Institution | Degree | Date" or "Degree | Date"
        ats_pattern = r'^(.+?)\s*\|\s*(.+?)\s*\|?\s*(\d{4}(?:\s*[-–]\s*\d{4})?)?\s*$'
        
        # Format 4: Compact single-line - "BS Computer Science, MIT, 2020"
        compact_pattern = r'^([A-Za-z\.\s]+)(?:in\s+)?([A-Za-z\s]+),\s*([A-Za-z\s]+(?:University|College|Institute)[A-Za-z\s]*),?\s*(\d{4})?'
        
        # Normalize section text - combine broken lines
        lines = section.split('\n')
        normalized_lines = []
        current_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_line:
                    normalized_lines.append(current_line)
                    current_line = ""
                continue
            
            # Check if this line is a continuation (starts with lowercase or special chars)
            if current_line and (line[0].islower() or line.startswith('·') or 
                                  line.startswith('(') or line.startswith('-')):
                current_line += " " + line
            else:
                if current_line:
                    normalized_lines.append(current_line)
                current_line = line
        
        if current_line:
            normalized_lines.append(current_line)
        
        current_entry = {}
        
        # Common section headers to skip
        education_headers = {
            'education', 'education:', 'academic background', 'academic credentials',
            'educational qualifications', 'qualifications', 'academic history',
            'educational background', 'degrees', 'academic qualifications',
            'education & training', 'education and training', 'academic record'
        }
        
        for line in normalized_lines:
            # Skip section header
            if line.lower().strip() in education_headers:
                continue
            
            line_lower = line.lower()
            
            # Format 2: Traditional degree format - "Bachelor of Science in Computer Science"
            traditional_match = re.match(traditional_degree_pattern, line, re.IGNORECASE)
            if traditional_match:
                degree_part = traditional_match.group(1).strip()
                field_part = traditional_match.group(2).strip() if traditional_match.group(2) else None
                
                if current_entry and current_entry.get('institution'):
                    entries.append(current_entry)
                    current_entry = {}
                
                current_entry['degree'] = degree_part
                if field_part:
                    current_entry['field_of_study'] = field_part
                continue
            
            # Format 3: ATS pipe-separated format - "MIT | BS Computer Science | 2020"
            ats_match = re.match(ats_pattern, line)
            if ats_match and '|' in line:
                parts = [p.strip() for p in line.split('|')]
                
                if len(parts) >= 2:
                    # Determine which part is institution vs degree
                    if any(kw in parts[0].lower() for kw in institution_keywords):
                        if current_entry and current_entry.get('institution'):
                            entries.append(current_entry)
                        current_entry = {
                            'institution': parts[0],
                            'degree': parts[1] if len(parts) > 1 else None,
                            'graduation_date': parts[2] if len(parts) > 2 else None
                        }
                    elif any(dk in parts[0].lower() for dk in ['bachelor', 'master', 'phd', 'diploma', 'associate', 'doctor']):
                        current_entry['degree'] = parts[0]
                        if len(parts) > 1 and any(kw in parts[1].lower() for kw in institution_keywords):
                            current_entry['institution'] = parts[1]
                        if len(parts) > 2:
                            current_entry['graduation_date'] = parts[2]
                    continue
            
            # Format 4: Compact single-line format
            compact_match = re.match(compact_pattern, line, re.IGNORECASE)
            if compact_match:
                if current_entry and current_entry.get('institution'):
                    entries.append(current_entry)
                current_entry = {
                    'degree': compact_match.group(1).strip(),
                    'field_of_study': compact_match.group(2).strip() if compact_match.group(2) else None,
                    'institution': compact_match.group(3).strip() if compact_match.group(3) else None,
                    'graduation_date': compact_match.group(4) if compact_match.group(4) else None
                }
                continue
            
            # Check if this is an institution line (contains institution keyword, no degree indicators)
            is_institution = (any(kw in line_lower for kw in institution_keywords) and 
                            not any(deg in line_lower for deg in ['diploma', 'bachelor', 'master', 'degree', 'certificate']))
            
            if is_institution:
                # Save previous entry if exists
                if current_entry and current_entry.get('institution'):
                    entries.append(current_entry)
                current_entry = {'institution': line.strip()}
                continue
            
            # LinkedIn format: "Diploma of Education, Logistics, Materials · (September 2022 - April 2023)"
            # Or: "Bachelor of Arts - BA, Political Science · (2018 - 2021)"
            linkedin_match = re.search(linkedin_degree_pattern, line.replace('·', '·'))
            if linkedin_match:
                degree_part = linkedin_match.group(1).strip()
                field_part = linkedin_match.group(2).strip()
                date_part = linkedin_match.group(3).strip()
                
                current_entry['degree'] = degree_part
                current_entry['field_of_study'] = field_part
                
                # Parse date range
                date_range_match = re.search(date_range_pattern, date_part, re.IGNORECASE)
                if date_range_match:
                    start_date = date_range_match.group(1).strip()
                    end_date = date_range_match.group(2).strip()
                    current_entry['graduation_date'] = f"{start_date} - {end_date}"
                else:
                    current_entry['graduation_date'] = date_part
                continue
            
            # Alternative: Check for "Degree, Field" pattern without explicit date formatting
            comma_split = line.split(',', 1)
            if len(comma_split) == 2:
                first_part = comma_split[0].strip()
                second_part = comma_split[1].strip()
                
                # Check if first part looks like a degree
                degree_keywords = ['diploma', 'bachelor', 'master', 'doctor', 'phd', 'certificate',
                                   'associate', 'degree', 'b.a', 'b.s', 'm.a', 'm.s', 'mba']
                if any(dk in first_part.lower() for dk in degree_keywords):
                    current_entry['degree'] = first_part
                    
                    # Extract field and dates from second part
                    # Remove date portion if present
                    date_range_match = re.search(date_range_pattern, second_part, re.IGNORECASE)
                    if date_range_match:
                        field = second_part[:date_range_match.start()].strip(' ·()-')
                        if field:
                            current_entry['field_of_study'] = field
                        start_date = date_range_match.group(1).strip()
                        end_date = date_range_match.group(2).strip()
                        current_entry['graduation_date'] = f"{start_date} - {end_date}"
                    else:
                        # Check for single date
                        single_match = re.search(single_date_pattern, second_part, re.IGNORECASE)
                        if single_match:
                            field = second_part[:single_match.start()].strip(' ·()-')
                            if field:
                                current_entry['field_of_study'] = field
                            current_entry['graduation_date'] = single_match.group(1)
                        else:
                            current_entry['field_of_study'] = second_part.strip(' ·()-')
                    continue
            
            # DOCX format: "Degree in Field | Date" (pipe separator)
            # Must check BEFORE standalone date check
            pipe_match = re.match(r'^(.+?)\s*\|\s*(\d{4}\s*[-–]\s*\d{4}|\d{4})\s*$', line)
            if pipe_match:
                degree_field_part = pipe_match.group(1).strip()
                date_part = pipe_match.group(2).strip()
                
                # Parse degree and field from "Diploma in Supply Chain & Logistics Management"
                # or "Bachelor of Arts (Political Science)"
                in_match = re.match(r'^(.+?)\s+in\s+(.+)$', degree_field_part, re.IGNORECASE)
                paren_match = re.match(r'^(.+?)\s*\(([^)]+)\)\s*$', degree_field_part)
                
                if in_match:
                    current_entry['degree'] = in_match.group(1).strip()
                    current_entry['field_of_study'] = in_match.group(2).strip()
                elif paren_match:
                    current_entry['degree'] = paren_match.group(1).strip()
                    current_entry['field_of_study'] = paren_match.group(2).strip()
                else:
                    current_entry['degree'] = degree_field_part
                
                current_entry['graduation_date'] = date_part.replace('–', '-')
                continue
            
            # Check for standalone date line (only if line is primarily just a date)
            date_range_match = re.search(date_range_pattern, line, re.IGNORECASE)
            if date_range_match:
                # Only use as standalone date if line is mostly the date (not part of other content)
                date_text = date_range_match.group(0)
                if len(date_text) > len(line) * 0.5:  # Date is more than half the line
                    start_date = date_range_match.group(1).strip()
                    end_date = date_range_match.group(2).strip()
                    current_entry['graduation_date'] = f"{start_date} - {end_date}"
                    continue
            
            # Check for GPA
            gpa_match = re.search(gpa_pattern, line, re.IGNORECASE)
            if gpa_match:
                gpa_value = gpa_match.group(1)
                gpa_scale = gpa_match.group(2) if gpa_match.group(2) else '4.0'
                current_entry['gpa'] = f"{gpa_value}/{gpa_scale}"
                continue
            
            # Fallback: Check for degree keywords in line (use word boundaries to avoid false matches)
            for abbr, full_name in degree_mapping.items():
                # Use word boundary matching to prevent "diploma" matching "ma" inside it
                abbr_pattern = r'\b' + re.escape(abbr) + r'\b'
                if re.search(abbr_pattern, line_lower) or full_name.lower() in line_lower:
                    if not current_entry.get('degree'):
                        current_entry['degree'] = full_name
                    break
        
        # Don't forget the last entry
        if current_entry and current_entry.get('institution'):
            entries.append(current_entry)
        
        # Set default None values for missing fields
        for entry in entries:
            entry.setdefault('institution', None)
            entry.setdefault('degree', None)
            entry.setdefault('field_of_study', None)
            entry.setdefault('graduation_date', None)
            entry.setdefault('gpa', None)
        
        return entries
    
    def _extract_section(self, text: str, section_names: List[str]) -> Optional[str]:
        """Extract a section from the resume text - handles multiple format styles
        
        Supports section headers in various formats:
        - ALL CAPS: "EXPERIENCE", "EDUCATION"
        - Title Case: "Work Experience", "Education"
        - With colons: "Experience:", "Skills:"
        - With underlines: "Experience" followed by dashes/underscores
        - With separators: "Experience & Training", "Skills / Competencies"
        - Numbered: "1. Experience", "2. Education"
        """
        # Build regex pattern for section headers (case-insensitive)
        section_pattern = '|'.join(re.escape(name) for name in section_names)
        
        # Extended list of all possible section headers
        all_sections = [
            'education', 'experience', 'skills', 'projects', 'certifications',
            'achievements', 'awards', 'publications', 'languages', 'interests',
            'references', 'summary', 'objective', 'profile', 'work history',
            'employment', 'professional', 'technical', 'personal', 'contact',
            'training', 'courses', 'coursework', 'research', 'volunteer',
            'extracurricular', 'activities', 'memberships', 'affiliations',
            'patents', 'presentations', 'conferences', 'honors', 'licenses',
            'additional', 'other', 'miscellaneous', 'professional development',
            'academic', 'career', 'qualifications', 'expertise', 'competencies',
            'highlights', 'overview', 'about', 'about me', 'bio', 'biography'
        ]
        end_sections = [s for s in all_sections if s.lower() not in [n.lower() for n in section_names]]
        end_pattern = '|'.join(re.escape(name) for name in end_sections)
        
        # Multiple patterns to find section headers
        section_start_patterns = [
            # Pattern 1: Standard with optional colon - "Experience:" or "Experience"
            rf'(?:^|\n)\s*({section_pattern})(?:\s*[&/,]\s*\w+)?\s*[:\n]',
            # Pattern 2: ALL CAPS - "EXPERIENCE" or "WORK EXPERIENCE"
            rf'(?:^|\n)\s*({section_pattern.upper()})(?:\s*[&/,]\s*\w+)?\s*[:\n]',
            # Pattern 3: With numbering - "1. Experience" or "I. Experience"
            rf'(?:^|\n)\s*(?:\d+\.|[IVX]+\.)\s*({section_pattern})\s*[:\n]?',
            # Pattern 4: With underline/separator on next line
            rf'(?:^|\n)\s*({section_pattern})\s*\n\s*[-=_]+\s*\n',
            # Pattern 5: In brackets or parentheses - "[Experience]" or "(Experience)"
            rf'(?:^|\n)\s*[\[\(]({section_pattern})[\]\)]\s*[:\n]?',
        ]
        
        start_match = None
        for pattern in section_start_patterns:
            start_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if start_match:
                break
        
        if not start_match:
            return None
        
        start_pos = start_match.end()
        
        # Find section end using similar flexible patterns
        end_patterns = [
            rf'(?:^|\n)\s*({end_pattern})(?:\s*[&/,]\s*\w+)?\s*[:\n]',
            rf'(?:^|\n)\s*({end_pattern.upper()})(?:\s*[&/,]\s*\w+)?\s*[:\n]',
            rf'(?:^|\n)\s*(?:\d+\.|[IVX]+\.)\s*({end_pattern})\s*[:\n]?',
            rf'(?:^|\n)\s*({end_pattern})\s*\n\s*[-=_]+\s*\n',
        ]
        
        end_match = None
        end_pos = None
        for pattern in end_patterns:
            match = re.search(pattern, text[start_pos:], re.IGNORECASE | re.MULTILINE)
            if match:
                if end_pos is None or match.start() < end_pos:
                    end_match = match
                    end_pos = match.start()
        
        if end_match:
            return text[start_pos:start_pos + end_match.start()]
        else:
            # Take rest of document or limit to reasonable length
            return text[start_pos:start_pos + 3000]
    
    def _extract_experience(self, doc, matches, phrase_matches) -> List[Dict[str, Any]]:
        """Extract work experience with comprehensive details - supports all resume formats
        
        Supported resume formats:
        - Chronological: Lists jobs in reverse chronological order
        - Functional: Groups experience by skills/functions
        - Combination: Mixes chronological and functional
        - Targeted: Customized for specific job
        - LinkedIn PDF: Exported from LinkedIn profile
        - ATS-friendly: Simple format with clear structure
        - Academic CV: Detailed with research/teaching positions
        - Creative: Modern layouts with unique formatting
        - European/Europass: Structured European format
        - Federal/Government: Detailed format for government jobs
        """
        experience = []
        text = doc.text
        
        # Skip words that are not actual employers
        skip_companies = {'linkedin', 'github', 'twitter', 'facebook', 'instagram', 'youtube',
                          'experience', 'education', 'skills', 'summary', 'team', 'page',
                          'projects', 'certifications', 'achievements', 'references', 'languages',
                          'and', 'the', 'with', 'from', 'for', 'dedicated', 'committed',
                          'experienced', 'professional', 'proven', 'track', 'record'}
        
        # Common company suffixes - extended for international companies
        company_suffixes = ['ltd', 'ltd.', 'inc', 'inc.', 'corp', 'corp.', 'llc', 'llp',
                           'company', 'co', 'co.', 'corporation', 'enterprises', 'solutions',
                           'technologies', 'tech', 'systems', 'services', 'group', 'partners',
                           'consulting', 'labs', 'studio', 'studios', 'agency', 'firm',
                           'designs', 'metals', 'industries', 'pvt', 'private', 'limited',
                           'gmbh', 's.a.', 'b.v.', 'n.v.', 'pty', 'plc', 'ag',
                           'healthcare', 'pharmaceuticals', 'media', 'entertainment', 'retail']
        
        # Job title keywords (standalone titles) - extended list
        job_title_keywords = [
            'supervisor', 'manager', 'engineer', 'developer', 'analyst', 'lead', 'director',
            'coordinator', 'specialist', 'consultant', 'architect', 'designer', 'scientist',
            'administrator', 'officer', 'executive', 'associate', 'intern', 'trainee',
            'founder', 'co-founder', 'owner', 'partner', 'president', 'vp', 'ceo', 'cto', 'cfo',
            'head', 'chief', 'principal', 'fellow', 'researcher', 'professor', 'lecturer',
            'assistant', 'staff', 'contractor', 'freelancer', 'technician', 'operator',
            'representative', 'agent', 'clerk', 'secretary', 'receptionist'
        ]
        
        # Duration pattern (X years Y months)
        duration_pattern = r'(\d+\s*years?\s*\d*\s*months?)'
        
        # Date range patterns - comprehensive list for all resume formats
        date_range_patterns = [
            # Format 1: Month Year - Month Year or Present (most common)
            r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–—~to]+\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow|[Oo]ngoing)',
            # Format 2: MM/YYYY - MM/YYYY or Present (ATS-friendly)
            r'(\d{1,2}/\d{4})\s*[-–—~to]+\s*(\d{1,2}/\d{4}|[Pp]resent|[Cc]urrent)',
            # Format 3: YYYY-MM - YYYY-MM (ISO-ish format)
            r'(\d{4}[-/]\d{1,2})\s*[-–—~to]+\s*(\d{4}[-/]\d{1,2}|[Pp]resent|[Cc]urrent)',
            # Format 4: Year only - 2020 - 2023 or 2020 - Present
            r'(\d{4})\s*[-–—~to]+\s*(\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)',
            # Format 5: European format DD/MM/YYYY or DD.MM.YYYY
            r'(\d{1,2}[./]\d{1,2}[./]\d{4})\s*[-–—~to]+\s*(\d{1,2}[./]\d{1,2}[./]\d{4}|[Pp]resent)',
            # Format 6: Abbreviated month 'Jan 2020 - Present'
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)['\.]?\s*\d{4})\s*[-–—~to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)['\.]?\s*\d{4}|[Pp]resent)",
            # Format 7: Season Year - 'Spring 2020 - Fall 2021'
            r'((?:Spring|Summer|Fall|Winter|Autumn)\s+\d{4})\s*[-–—~to]+\s*((?:Spring|Summer|Fall|Winter|Autumn)\s+\d{4}|[Pp]resent)',
            # Format 8: With parentheses - (2020 - 2023)
            r'\((\d{4})\s*[-–—]\s*(\d{4}|[Pp]resent)\)',
        ]
        
        # Try multiple parsing strategies for different resume formats
        # Strategy 1: Try LinkedIn-style parsing first
        # LinkedIn PDFs often have: "Company Name\n Duration\n Title\n Date Range\n Location"
        linkedin_entries = self._parse_linkedin_experience(text, company_suffixes, job_title_keywords, 
                                                            duration_pattern, date_range_patterns, skip_companies)
        if linkedin_entries:
            experience.extend(linkedin_entries)
        
        # Strategy 2: Section-based extraction for standard resume formats
        # Expanded section names to catch all variations
        exp_section = self._extract_section(text, [
            'experience', 'work experience', 'employment history', 'work history', 
            'professional experience', 'career history', 'employment',
            'relevant experience', 'related experience', 'professional background',
            'career summary', 'positions held', 'job history', 'career experience',
            'working experience', 'industry experience', 'practical experience',
            'professional history', 'employment record', 'career path',
            'appointments', 'positions', 'roles'
        ])
        
        if exp_section:
            section_entries = self._parse_experience_section_v2(exp_section, company_suffixes, 
                                                                 job_title_keywords, skip_companies)
            # Merge entries - add section entries that aren't already in experience
            existing_companies = {((e.get('company') or '').lower(), e.get('position', '') or '') for e in experience}
            for entry in section_entries:
                company = entry.get('company') or ''
                position = entry.get('position') or ''
                entry_key = (company.lower(), position)
                # Add if not a duplicate and has a company name
                if entry_key not in existing_companies and company:
                    experience.append(entry)
                    existing_companies.add(entry_key)
        
        # Strategy 3: Try functional resume format
        # Functional resumes group achievements by skill area instead of by job
        if not experience:
            functional_entries = self._parse_functional_resume(text, company_suffixes, 
                                                                job_title_keywords, skip_companies)
            if functional_entries:
                experience.extend(functional_entries)
        
        # Strategy 4: Use spaCy ORG entities as fallback for companies
        if not experience:
            seen_orgs = set()
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    org_name = ent.text.strip()
                    org_lower = org_name.lower()
                    if (org_lower not in skip_companies and 
                        org_lower not in seen_orgs and 
                        len(org_name) > 3 and
                        not any(kw in org_lower for kw in ['university', 'college', 'school', 'institute'])):
                        seen_orgs.add(org_lower)
                        experience.append({
                            "company": org_name,
                            "position": None,
                            "start_date": None,
                            "end_date": None,
                            "duration": None,
                            "location": None,
                            "description": None,
                            "responsibilities": []
                        })
        
        # Deduplicate and clean - remove entries with very short or fragment company names
        seen = set()
        cleaned_experience = []
        
        # Words that indicate garbage/fragment text in company names
        garbage_words = {'ensure', 'provide', 'develop', 'maintain', 'create', 'implement',
                         'utilize', 'manage', 'support', 'coordinate', 'facilitate', 'foster',
                         'accuracy', 'efficiency', 'effectiveness', 'productivity', 'quality',
                         'operational', 'strategic', 'professional', 'excellent', 'timely',
                         'various', 'multiple', 'different', 'specific', 'relevant'}
        
        for exp in experience:
            company = (exp.get('company') or '').strip()
            position = (exp.get('position') or '').strip()
            
            # Clean company name - replace newlines with spaces and normalize
            company = ' '.join(company.split())
            
            # Skip LinkedIn summary entries - these appear as "(X months) at Company" 
            # and get parsed with company starting with duration or "at"
            if re.match(r'^\(\d+\s*(?:years?|months?)', company):
                continue
            if company.lower().startswith('at '):
                continue
            # Skip if position starts with duration pattern (indicates summary entry)
            if position and re.match(r'^\(\d+\s*(?:years?|months?)', position):
                continue
            
            # Skip if company name is too short or looks like a fragment
            if company:
                company_lower = company.lower()
                company_words = company_lower.split()
                
                if len(company) < 4:
                    continue
                if any(company_lower.startswith(skip) for skip in skip_companies):
                    continue
                # Skip fragmented text (ends with prepositions/articles)
                if company_lower.endswith(('co', 'with', 'from', 'and', 'the', 'a', 'an', 'to', 'for')):
                    continue
                # Skip if any word is a garbage word (indicates fragment text)
                if any(word in garbage_words for word in company_words):
                    continue
                # Skip if company contains only lowercase words (not proper names)
                original_words = company.split()
                if len(original_words) >= 2 and all(w[0].islower() for w in original_words if w):
                    continue
                # Skip if company contains "Present" (parsing error)
                if 'present' in company_lower:
                    # Try to extract just the company name after "Present "
                    if company_lower.startswith('present '):
                        company = company[8:].strip()
                        if len(company) < 4:
                            continue
                    else:
                        continue
                # Skip if position is merged into company name (e.g., "Supervisor Aluminum Window...")
                # Check if company starts with a job title
                
                # First check for multi-word positions like "Shipping Associate" (MUST be checked before single-word)
                two_word_positions = [
                    'shipping associate', 'production supervisor', 'warehouse manager',
                    'operations manager', 'sales representative', 'marketing manager',
                    'project coordinator', 'business analyst', 'quality inspector',
                    'customer service', 'technical support', 'general manager',
                    'assistant manager', 'senior developer', 'junior developer',
                    'team lead', 'shift supervisor', 'floor manager', 'area manager',
                    'production associate', 'warehouse associate', 'sales associate',
                    'customer associate', 'service representative', 'account manager',
                    'hr manager', 'finance manager', 'it manager', 'logistics coordinator',
                    'supply chain', 'quality assurance', 'quality control', 'data analyst'
                ]
                position_found = False
                for two_word_pos in two_word_positions:
                    if company_lower.startswith(two_word_pos + ' '):
                        remaining = company[len(two_word_pos)+1:].strip()
                        if remaining and len(remaining) > 3:
                            exp['position'] = exp.get('position') or company[:len(two_word_pos)+1].strip().title()
                            company = remaining
                            position_found = True
                        break
                
                # Only check single-word titles if no two-word position was found
                if not position_found:
                    # Extended list of position keywords that might be merged with company names
                    extended_title_keywords = job_title_keywords + [
                        'shipping', 'production', 'warehouse', 'operations', 'sales', 'marketing',
                        'finance', 'accounting', 'customer', 'technical', 'project', 'business',
                        'quality', 'senior', 'junior', 'head', 'chief', 'assistant', 'general',
                        'regional', 'national', 'global', 'deputy', 'vice', 'staff', 'line'
                    ]
                    
                    for title_kw in extended_title_keywords:
                        if company_lower.startswith(title_kw + ' '):
                            # Extract position and company separately
                            remaining = company[len(title_kw)+1:].strip()
                            if remaining and len(remaining) > 3:
                                exp['position'] = exp.get('position') or company[:len(title_kw)+1].strip().title()
                                company = remaining
                            break
                
                # Update the cleaned company name
                exp['company'] = company
            
            # Clean position name too
            if position:
                position = ' '.join(position.split())
                exp['position'] = position
            
            company_key = company.lower() if company else ''
            position_key = position.lower() if position else ''
            full_key = (company_key, position_key)
            
            # Skip entries without positions if we already have any entry for this company with a position
            if company_key and not position_key:
                # Check if we already have an entry with this company AND a position
                has_better_entry = any(
                    (existing.get('company') or '').lower() == company_key and existing.get('position')
                    for existing in cleaned_experience
                )
                if has_better_entry:
                    continue  # Skip this incomplete entry
            
            if full_key != ('', '') and full_key not in seen:
                seen.add(full_key)
                # Also track company-only key when adding an entry WITH a position
                # This allows us to filter out later no-position entries for same company
                if position_key:
                    # Remove any existing entry for this company without a position
                    cleaned_experience = [
                        e for e in cleaned_experience
                        if not ((e.get('company') or '').lower() == company_key and not e.get('position'))
                    ]
                cleaned_experience.append(exp)
        
        return cleaned_experience[:10]  # Limit to 10 experiences
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract responsibilities from text block - handles various formats"""
        responsibilities = []
        seen_descriptions = set()  # Track descriptions to avoid duplicates
        
        # Clean text - normalize line breaks and remove page markers
        text = re.sub(r'\s*\n\s*', ' ', text)
        text = re.sub(r'\s*Page\s+\d+\s+of\s+\d+\s*', ' ', text, flags=re.IGNORECASE)
        
        # Pattern 1: "Label: Description" format (LinkedIn style)
        # Examples: "Team Leadership: Supervise and mentor a diverse team..."
        responsibility_labels = [
            'Team Leadership', 'Production Management', 'Asset Management',
            'CNC Operation', 'Strategic Planning', 'Enterprise Resource Planning', 'ERP',
            'Employee Management', 'Reporting', 'Operational Oversight', 'Best Practices',
            'Problem Solving', 'Performance Monitoring', 'Logistics Management', 
            'Transportation Coordination', 'Material Receipt', 'Process Improvement',
            'Business Development', 'Team Management', 'Customer Engagement',
            'Project Management', 'Quality Assurance', 'Quality Control', 'Data Analysis',
            'Client Relations', 'Vendor Management', 'Budget Management', 'Risk Management',
            'Sales', 'Marketing', 'Operations', 'Training', 'Development', 'Research',
            'Implementation', 'Analysis', 'Communication', 'Collaboration',
            'Material Receipt and Tracking', 'Best Practices Promotion',
            'Software Development', 'Technical Support', 'Customer Service',
            'Financial Analysis', 'Product Development', 'Supply Chain', 'Inventory Management'
        ]
        
        # Build a pattern that captures text between labels
        # First, find all label positions in the text
        label_positions = []
        for label in responsibility_labels:
            for match in re.finditer(rf'\b{re.escape(label)}:', text, re.IGNORECASE):
                label_positions.append((match.start(), match.end(), label))
        
        # Sort by position
        label_positions.sort(key=lambda x: x[0])
        
        # Extract content between each label and the next one
        for i, (start, end, label) in enumerate(label_positions):
            # Find the end of this responsibility (start of next label or end of text chunk)
            if i + 1 < len(label_positions):
                next_start = label_positions[i + 1][0]
            else:
                next_start = len(text)
            
            # Get the description text
            description = text[end:next_start].strip()
            
            # Clean up the description
            # Stop at Page markers or section headers
            description = re.sub(r'\s*Page\s+\d+.*$', '', description)
            description = re.sub(r'\s*(?:Education|Experience|Skills|Summary)\s*$', '', description, flags=re.IGNORECASE)
            
            # Get just the first sentence or up to 200 chars
            period_match = re.search(r'^([^.]+\.)', description)
            if period_match and len(period_match.group(1)) >= 20:
                description = period_match.group(1).strip()
            else:
                # Take first 200 chars if no period found
                description = description[:200].strip()
                if len(description) > 20:
                    # Try to end at last complete word
                    last_space = description.rfind(' ')
                    if last_space > 20:
                        description = description[:last_space]
            
            if len(description) < 20:
                continue
            
            # Normalize for duplicate check
            desc_normalized = description.lower()[:60]
            if desc_normalized in seen_descriptions:
                continue
            seen_descriptions.add(desc_normalized)
            
            full_responsibility = f"{label}: {description}"
            responsibilities.append(full_responsibility)
        
        # Pattern 2: Bullet point format (• or - prefixed)
        bullet_pattern = r'[•\-\*\▪\▸\→\›\○\●]\s*([A-Z][^•\-\*\▪\▸\→\›\○\●]+?)(?=[•\-\*\▪\▸\→\›\○\●]|$)'
        for match in re.finditer(bullet_pattern, text):
            resp = match.group(1).strip()
            if resp and len(resp) > 20:
                desc_normalized = resp.lower()[:60]
                if desc_normalized not in seen_descriptions:
                    seen_descriptions.add(desc_normalized)
                    responsibilities.append(resp)
        
        # Post-processing: Filter out false positives
        # Remove entries that start with dates, locations, or "Present"
        filtered = []
        for resp in responsibilities:
            # Skip if it starts with date patterns or locations
            if re.match(r'^(?:Present|January|February|March|April|May|June|July|August|September|October|November|December)\s', resp, re.IGNORECASE):
                continue
            if re.match(r'^\d{4}\s', resp):
                continue
            # Skip if starts with a location (City, State pattern)
            if re.match(r'^[A-Z][a-z]+,?\s+[A-Z][a-z]+,?\s+(?:Canada|India|USA|Ontario|[A-Z]{2})\s', resp):
                continue
            # Skip if it contains multiple responsibility labels concatenated (noise from PDF parsing)
            label_count = sum(1 for label in responsibility_labels if f'{label}:' in resp)
            if label_count > 1:
                continue
            # Skip if it looks like education entry (contains degree + university pattern)
            education_patterns = [
                r'\b(?:BA|BS|MA|MS|MBA|PhD|BSc|MSc|BBA|MCA|BCA|BEng|MEng|BTech|MTech)\b[,\s]+.*(?:University|College|Institute)',
                r'\b(?:Bachelor|Master|Doctor|Diploma|Degree)\b[,\s]+.*(?:University|College|Institute)',
                r'\b(?:University|College|Institute)\b[,\s]+\d{4}',
                r'^[A-Z][A-Za-z]+\s+(?:of\s+)?(?:Arts|Science|Engineering|Commerce|Technology)\s+.*(?:University|College)',
            ]
            is_education = False
            for edu_pattern in education_patterns:
                if re.search(edu_pattern, resp, re.IGNORECASE):
                    is_education = True
                    break
            if is_education:
                continue
            filtered.append(resp)
        
        # Limit to top 8 responsibilities per entry
        return filtered[:8]
    
    def _parse_functional_resume(self, text: str, company_suffixes: List[str],
                                  job_title_keywords: List[str], skip_companies: set) -> List[Dict[str, Any]]:
        """Parse functional/skills-based resume format
        
        Functional resumes group experience by skill areas rather than chronologically.
        Common in career changers, those with gaps, or emphasizing transferable skills.
        
        Format typically looks like:
        - LEADERSHIP EXPERIENCE
          - Achievement 1 at Company A
          - Achievement 2 at Company B
        - TECHNICAL SKILLS
          - Built X system at Company C
          
        This method extracts company names and positions from within achievement bullets.
        """
        entries = []
        
        # Look for functional sections (skill-based headers)
        functional_sections = [
            'leadership experience', 'management experience', 'technical experience',
            'project management', 'team leadership', 'customer service experience',
            'sales experience', 'marketing experience', 'administrative experience',
            'relevant accomplishments', 'key accomplishments', 'selected achievements',
            'professional accomplishments', 'career highlights', 'key achievements'
        ]
        
        for section_name in functional_sections:
            section = self._extract_section(text, [section_name])
            if section:
                # Look for company mentions within this section
                # Pattern: "at Company Name" or "for Company Name" or "with Company Name"
                company_mentions = re.findall(
                    r'(?:at|for|with)\s+([A-Z][A-Za-z\s]+(?:' + 
                    '|'.join(re.escape(s) for s in company_suffixes) + r'))',
                    section,
                    re.IGNORECASE
                )
                
                for company in company_mentions:
                    company = company.strip()
                    if company.lower() not in skip_companies and len(company) > 3:
                        # Check if we already have this company
                        if not any(e.get('company', '').lower() == company.lower() for e in entries):
                            entries.append({
                                "company": company,
                                "position": None,  # Functional resumes often don't specify position per company
                                "start_date": None,
                                "end_date": None,
                                "duration": None,
                                "location": None,
                                "description": section_name.title(),
                                "responsibilities": []
                            })
                
                # Also look for "Company Name (Year - Year)" pattern
                company_date_pattern = re.findall(
                    r'([A-Z][A-Za-z\s]+)\s*\((\d{4})\s*[-–]\s*(\d{4}|[Pp]resent)\)',
                    section
                )
                for company, start, end in company_date_pattern:
                    company = company.strip()
                    if company.lower() not in skip_companies and len(company) > 3:
                        if not any(e.get('company', '').lower() == company.lower() for e in entries):
                            entries.append({
                                "company": company,
                                "position": None,
                                "start_date": start,
                                "end_date": end,
                                "duration": None,
                                "location": None,
                                "description": section_name.title(),
                                "responsibilities": []
                            })
        
        # Also look for a "Work History" or "Employment Summary" section 
        # (common at end of functional resumes, just lists companies and dates)
        employment_summary = self._extract_section(text, [
            'employment summary', 'work history', 'employment record', 
            'career timeline', 'job history'
        ])
        
        if employment_summary:
            # Pattern: "Company Name, Position, Dates" or "Company Name | Position | Dates"
            summary_pattern = re.findall(
                r'([A-Z][A-Za-z\s&,\.]+?)(?:\s*[,|]\s*)([A-Za-z\s]+)?(?:\s*[,|]\s*)?' +
                r'(\d{4}\s*[-–]\s*(?:\d{4}|[Pp]resent))',
                employment_summary
            )
            
            for company, position, dates in summary_pattern:
                company = company.strip().rstrip(',|')
                position = position.strip() if position else None
                
                if company.lower() not in skip_companies and len(company) > 3:
                    # Parse dates
                    date_match = re.match(r'(\d{4})\s*[-–]\s*(\d{4}|[Pp]resent)', dates)
                    start_date = date_match.group(1) if date_match else None
                    end_date = date_match.group(2) if date_match else None
                    
                    # Check for duplicate
                    if not any(e.get('company', '').lower() == company.lower() for e in entries):
                        entries.append({
                            "company": company,
                            "position": position,
                            "start_date": start_date,
                            "end_date": end_date,
                            "duration": None,
                            "location": None,
                            "description": None,
                            "responsibilities": []
                        })
        
        return entries
    
    def _parse_linkedin_experience(self, text: str, company_suffixes: List[str], 
                                    job_title_keywords: List[str], duration_pattern: str,
                                    date_range_patterns: List[str], skip_companies: set) -> List[Dict[str, Any]]:
        """Parse experience using company-suffix based detection
        
        This method works well for resumes that have explicit company suffixes like:
        - LinkedIn PDF exports
        - Formal/corporate resumes
        - Resumes with clear company names (e.g., "Google Inc.", "Microsoft Corp")
        
        Also handles:
        - Startup/founder entries
        - Freelance/consulting entries
        - Internship entries
        """
        entries = []
        
        # Remove LinkedIn summary section entries that duplicate experience
        # These appear as "(X years Y months) at Company Name" format at the end of the PDF
        # Pattern 1: "(11 months) at Company Name | Date - Date"
        summary_pattern1 = r'\(\d+\s*(?:years?|months?)(?:\s*\d+\s*(?:months?))?\)\s+at\s+[^\n]+'
        text = re.sub(summary_pattern1, '', text, flags=re.IGNORECASE)
        
        # Pattern 2: Just "(X months)" at start of a line followed by company
        summary_pattern2 = r'^\s*\(\d+\s*(?:years?|months?)\)\s+at\s+'
        text = re.sub(summary_pattern2, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Also remove fragmented bullet points from summary section
        # These start with bullet and have truncated content
        fragment_pattern = r'•\s*[a-z][^•\n]{5,100}\.\.\.'
        text = re.sub(fragment_pattern, '', text, flags=re.IGNORECASE)
        
        # Extended skip patterns for fragments
        extended_skip = skip_companies | {
            'utilize', 'provide', 'develop', 'ensure', 'implement', 'maintain',
            'facilitating', 'fostering', 'managing', 'coordinating', 'supporting',
            'systems', 'solutions', 'timely', 'resolutions', 'feedback'
        }
        
        # Extended company suffixes to catch more company types
        extended_company_suffixes = company_suffixes + [
            'gmbh', 's.a.', 'b.v.', 'n.v.', 'pty', 'plc', 's.r.l.', 'a.s.', 'ag',
            'healthcare', 'bank', 'insurance', 'pharmaceuticals', 'media',
            'foundation', 'hospital', 'clinic', 'university', 'school'
        ]
        
        # Company pattern with flexible suffix matching
        # Pattern: "Company Name Ltd\n2 years 3 months\nJob Title\nMonth Year - Present"
        company_pattern = r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:' + '|'.join(re.escape(s) for s in extended_company_suffixes) + r'))\b\s*\n?\s*(' + duration_pattern + r')?'
        
        for match in re.finditer(company_pattern, text, re.IGNORECASE):
            company = match.group(1).strip()
            duration = match.group(2).strip() if match.group(2) else None
            
            # Clean company name - remove section header prefixes (Experience, Education, etc.)
            # The regex can match across newlines, so we need to clean these up
            section_headers = ['experience', 'education', 'skills', 'certifications', 'about', 'summary']
            for header in section_headers:
                # Check for header followed by space or newline
                if company.lower().startswith(header + ' '):
                    company = company[len(header)+1:].strip()
                    break
                elif company.lower().startswith(header + '\n'):
                    company = company[len(header)+1:].strip()
                    break
                # Also check with just header at start (newline might be stripped)
                elif '\n' in company:
                    lines = company.split('\n')
                    if lines[0].lower().strip() in section_headers:
                        company = '\n'.join(lines[1:]).strip()
                        break
            
            # Final cleanup - replace any remaining newlines with spaces
            company = ' '.join(company.split())
            
            # Validate company name - more strict
            if not company or len(company) < 5:
                continue
            company_words = company.lower().split()
            # Skip if any word is in extended skip list
            if any(word in extended_skip for word in company_words):
                continue
            # Skip if it's just a suffix
            if company.lower() in company_suffixes:
                continue
            
            # Find the end of this entry - either next company or section header
            # Search for next company name in the text
            remaining_text = text[match.end():]
            
            # Find next company pattern (to limit scope)
            # More strict pattern - require capitalized proper name structure
            # Exclude patterns that start with common verbs
            strict_company_suffixes = ['ltd', 'ltd.', 'inc', 'inc.', 'corp', 'corp.', 'llc', 'llp',
                                       'corporation', 'enterprises', 'metals', 'industries', 
                                       'pvt', 'private', 'limited']
            next_company_match = re.search(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(' + '|'.join(re.escape(s) for s in strict_company_suffixes) + r')\b',
                remaining_text[500:],  # Skip more chars to avoid false matches
                re.IGNORECASE
            )
            
            # Also look for role-based entries (e.g., "Dineing Founder", "Swiggy Internship")
            next_role_match = re.search(
                r'\b([A-Z][a-z]+)\s+(Founder|Co-Founder|Internship|Intern)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
                remaining_text[200:],  # Start checking earlier for these
                re.IGNORECASE
            )
            
            # Find section headers that mark end of experience section
            # Use more specific patterns to avoid matching "Top Skills" etc.
            section_match = re.search(
                r'\b(?:Education\s+[A-Z]|Education\s*$|'
                r'Projects?\s+[A-Z]|Certifications?\s+[A-Z]|'
                r'Languages?\s+(?:Punjabi|Hindi|English|Spanish|French|[A-Z][a-z]+\s+\()|'
                r'References\s*$)\b', 
                remaining_text[500:],  # Skip at least 500 chars to avoid false matches
                re.IGNORECASE
            )
            
            # Determine end position
            end_pos = 2500  # Default max - increased to capture more responsibilities
            if next_company_match:
                end_pos = min(end_pos, 500 + next_company_match.start())
            if next_role_match:
                end_pos = min(end_pos, 200 + next_role_match.start())
            if section_match:
                end_pos = min(end_pos, 500 + section_match.start())
            
            following_text = remaining_text[:end_pos]
            
            position = None
            start_date = None
            end_date = None
            location = None
            
            # Find job title - look for title immediately after company/duration
            # Pattern: "Job Title Month Year" or "Job Title (duration)"
            # Look in first 200 chars for the title
            title_search_area = following_text[:200]
            
            # Try more specific patterns first
            # Pattern 1: Multi-word title followed by date
            specific_title_match = re.search(
                r'^[^a-zA-Z]*([A-Z][a-zA-Z&\s]+?(?:Associate|Manager|Engineer|Developer|Supervisor|Lead|Director|Coordinator|Specialist|Analyst|Consultant|Officer|Executive|Intern))\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
                title_search_area, 
                re.IGNORECASE
            )
            if specific_title_match:
                position = specific_title_match.group(1).strip()
            else:
                # Pattern 2: Simple title keyword
                for title_kw in job_title_keywords:
                    title_pattern = rf'\b([A-Za-z\s&]*{title_kw})\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                    title_match = re.search(title_pattern, title_search_area, re.IGNORECASE)
                    if title_match:
                        position = title_match.group(1).strip()
                        break
            
            # Clean up the position if found
            if position:
                position = re.sub(r'\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b.*$', '', position, flags=re.IGNORECASE).strip()
                if len(position) > 50:
                    position = None
            
            # Find date range
            for date_pattern in date_range_patterns:
                date_match = re.search(date_pattern, following_text)
                if date_match:
                    start_date = date_match.group(1)
                    end_date = date_match.group(2)
                    break
            
            # Find location
            location_pattern = r'([A-Z][a-z]+,\s*(?:[A-Z][a-z]+,\s*)?(?:Ontario|Quebec|British Columbia|Alberta|Canada|USA|India|[A-Z]{2}))'
            loc_match = re.search(location_pattern, following_text)
            if loc_match:
                location = loc_match.group(1)
            
            # Extract responsibilities from the following text
            responsibilities = self._extract_responsibilities(following_text)
            
            entries.append({
                "company": company,
                "position": position,
                "start_date": start_date,
                "end_date": end_date,
                "duration": duration,
                "location": location,
                "description": None,
                "responsibilities": responsibilities
            })
        
        # Pattern 2: Catch entries without company suffixes (e.g., "Dineing Founder", "Swiggy Internship")
        # These have format: "Company Name Job_Title Month Year - Month Year"
        role_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(Founder|Co-Founder|Internship|Intern|Freelance|Consultant)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\s*[-–—]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|[Pp]resent)',
        ]
        
        for pattern in role_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                company = match.group(1).strip()
                position = match.group(2).strip()
                start_date = match.group(3).strip()
                end_date = match.group(4).strip()
                
                # Skip if already captured
                if any(e['company'].lower() == company.lower() for e in entries):
                    continue
                
                # Get following text for responsibilities
                remaining_text = text[match.end():]
                
                # Find next entry or section
                next_entry = re.search(
                    r'\b[A-Z][a-z]+\s+(?:Founder|Internship|Intern|Ltd|Inc|Corp|Metals)\b',
                    remaining_text[50:],
                    re.IGNORECASE
                )
                section_end = re.search(r'\b(?:Education|Skills|Projects)\b', remaining_text, re.IGNORECASE)
                
                end_pos = 1500
                if next_entry:
                    end_pos = min(end_pos, 50 + next_entry.start())
                if section_end:
                    end_pos = min(end_pos, section_end.start())
                
                following_text = remaining_text[:end_pos]
                
                # Extract location - improved pattern for various formats
                location = None
                # Pattern 1: Full location with city, state/region, country
                loc_match = re.search(r'(?:\d+\s*months?\))\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+)*,?\s*(?:India|Canada|USA|UK))', following_text)
                if loc_match:
                    location = loc_match.group(1).strip()
                else:
                    # Pattern 2: Just country after duration
                    loc_match = re.search(r'(?:\d+\s*months?\))\s*(India|Canada|USA|UK)\b', following_text)
                    if loc_match:
                        location = loc_match.group(1)
                
                # Extract responsibilities
                responsibilities = self._extract_responsibilities(following_text)
                
                # Extract duration if present
                duration = None
                dur_match = re.search(r'\((\d+\s*months?)\)', following_text)
                if dur_match:
                    duration = dur_match.group(1)
                
                entries.append({
                    "company": company,
                    "position": position,
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration": duration,
                    "location": location,
                    "description": None,
                    "responsibilities": responsibilities
                })
        
        return entries
    
    def _parse_experience_section_v2(self, section: str, company_suffixes: List[str],
                                      job_title_keywords: List[str], skip_companies: set) -> List[Dict[str, Any]]:
        """Parse experience entries from a section - handles multiple resume formats
        
        Supported formats:
        - Traditional chronological: Company -> Title -> Dates -> Responsibilities
        - Reverse chronological: Title -> Company -> Dates
        - ATS-friendly: Title | Company | Dates (pipe separated)
        - Functional: Grouped by skill/function
        - Combination: Mixed format
        - Academic CV: Detailed with multiple positions per institution
        - European/Europass: Structured format
        - Compact: Single-line entries
        """
        entries = []
        lines = [l.strip() for l in section.split('\n') if l.strip()]
        
        current_company = None
        current_location = None
        current_entry = None
        responsibilities = []
        
        # Skip lines that contain fragment phrases (action verbs that start responsibilities)
        fragment_words = {'utilize', 'provide', 'develop', 'ensure', 'implement', 
                         'maintain', 'facilitating', 'fostering', 'managing',
                         'coordinated', 'oversee', 'lead and', 'managed daily',
                         'conduct', 'conducted', 'developed', 'assisted', 'launched',
                         'recruited', 'monitored', 'oversaw', 'lead ', 'mentor',
                         'successfully', 'improved', 'increased', 'reduced', 'created',
                         'designed', 'built', 'established', 'achieved', 'delivered'}
        
        # Location pattern - must contain comma (city, state) OR be just a country name
        # Extended to support more international locations
        location_pattern = re.compile(
            r'^(?:'
            # City, Province/State, Country pattern
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z][a-zA-Z\s]+(?:,\s*(?:Canada|USA|US|India|UK|Australia|Germany|France|China|Japan|Singapore|UAE|Netherlands|Spain|Italy|Brazil|Mexico|Ireland|Switzerland|Sweden|Norway|Denmark|Finland|Belgium|Austria|New Zealand|South Korea|Taiwan|Hong Kong|Remote))?'
            r'|'
            # Just the country/state/region name alone
            r'(?:India|Canada|USA|US|UK|Australia|Germany|France|China|Japan|Singapore|UAE|Netherlands|Spain|Italy|Brazil|Mexico|Ireland|Switzerland|Sweden|Norway|Remote|Ontario|Quebec|British Columbia|Alberta|California|New York|Texas|Washington|Massachusetts|Colorado|Georgia|Florida|Illinois|Pennsylvania)'
            r'|'
            # Remote/Hybrid work patterns
            r'(?:Remote|Hybrid|On-site|Onsite|Work from Home|WFH)(?:\s*[-–/]\s*[A-Z][a-z]+)?'
            r')\s*$',
            re.IGNORECASE
        )
        
        # Comprehensive date patterns for different formats
        date_patterns = [
            # Format 1: Title | Month Year – Present (pipe separator, common in ATS)
            re.compile(
                r'^(.+?)\s*[|\|]\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
                r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–—~to]+\s*'
                r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
                r'Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|[Pp]resent|[Cc]urrent)',
                re.IGNORECASE
            ),
            # Format 2: Title | MM/YYYY - MM/YYYY (numeric dates)
            re.compile(
                r'^(.+?)\s*[|\|]\s*(\d{1,2}/\d{4})\s*[-–—~to]+\s*(\d{1,2}/\d{4}|[Pp]resent|[Cc]urrent)',
                re.IGNORECASE
            ),
            # Format 3: Title | YYYY - YYYY (year only)
            re.compile(
                r'^(.+?)\s*[|\|]\s*(\d{4})\s*[-–—~to]+\s*(\d{4}|[Pp]resent|[Cc]urrent)',
                re.IGNORECASE
            ),
            # Format 4: Title (Month Year – Present) - parentheses around dates
            re.compile(
                r'^(.+?)\s*\(\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
                r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–—~to]+\s*'
                r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
                r'Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|[Pp]resent|[Cc]urrent)\s*\)',
                re.IGNORECASE
            ),
            # Format 5: Month Year – Present followed by Title (date first)
            re.compile(
                r'^((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
                r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–—~to]+\s*'
                r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
                r'Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|[Pp]resent|[Cc]urrent)\s*[:\|]?\s*(.+?)$',
                re.IGNORECASE
            ),
        ]
        
        # Standalone date range pattern (for lines that are just dates)
        standalone_date_pattern = re.compile(
            r'^((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
            r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{1,2}/\d{4}|\d{4})\s*'
            r'[-–—~to]+\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
            r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{1,2}/\d{4}|\d{4}|[Pp]resent|[Cc]urrent)\s*$',
            re.IGNORECASE
        )
        
        # Company with date range on same line pattern
        company_date_pattern = re.compile(
            r'^([A-Z][A-Za-z\s&,\.]+?)\s*[,|\|]?\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
            r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{1,2}/\d{4}|\d{4})\s*'
            r'[-–—~to]+\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
            r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{1,2}/\d{4}|\d{4}|[Pp]resent|[Cc]urrent)',
            re.IGNORECASE
        )
        
        def is_company_line(line: str) -> bool:
            """Detect if a line is a company name - handles multiple resume formats"""
            line_lower = line.lower()
            
            # Skip lines that are too long (> 8 words) - likely descriptions
            words = line.split()
            if len(words) > 8:
                return False
            
            # Skip if starts with common action verbs (responsibilities)
            action_starts = ('conduct', 'develop', 'assist', 'launch', 'recruit', 'monitor',
                           'oversee', 'lead', 'mentor', 'successfully', 'improve', 'increase',
                           'reduce', 'create', 'design', 'build', 'establish', 'achieve',
                           'deliver', 'manage', 'maintain', 'implement', 'provide', 'utilize',
                           'coordinate', 'support', 'facilitate', 'foster', 'responsible',
                           'collaborated', 'spearheaded', 'initiated', 'streamlined', 'optimized',
                           'analyzed', 'executed', 'oversaw', 'drove', 'led', 'supervised')
            if line_lower.startswith(action_starts):
                return False
            
            # Check for company suffix FIRST (before period check) 
            # Extended list of company suffixes for international companies
            extended_suffixes = company_suffixes + [
                'gmbh', 's.a.', 'b.v.', 'n.v.', 'pty', 'plc', 's.r.l.', 'a.s.', 'ag',
                'healthcare', 'pharmaceuticals', 'bank', 'insurance', 'media', 'publishing',
                'entertainment', 'retail', 'logistics', 'manufacturing', 'automotive',
                'aerospace', 'defense', 'energy', 'utilities', 'telecom', 'wireless',
                'foundation', 'association', 'institute', 'laboratory', 'research',
                'network', 'platform', 'digital', 'interactive', 'creative', 'global'
            ]
            has_suffix = any(suffix in line_lower for suffix in extended_suffixes)
            if has_suffix and len(line) > 5:
                return True
            
            # Check if line contains company indicator patterns
            # E.g., "Company Name, Location" or "Company Name - Location"
            company_with_location = re.match(
                r'^([A-Z][A-Za-z\s&\.]+)(?:\s*[,\-–|]\s*[A-Z][a-z]+(?:,\s*[A-Z]{2,})?)?$',
                line
            )
            if company_with_location and len(words) <= 6:
                potential_company = company_with_location.group(1).strip()
                if len(potential_company) > 3 and not any(kw in potential_company.lower() for kw in job_title_keywords):
                    return True
            
            # For non-suffix lines: skip lines that end with a period (likely a sentence/description)
            if line.endswith('.') and not any(suff in line_lower for suff in ['inc.', 'ltd.', 'corp.', 'co.']):
                return False
            
            # Check if line is a capitalized name (1-5 words, no job title keywords, no dates)
            if 1 <= len(words) <= 5:
                # Must start with capital letter
                if not line[0].isupper():
                    return False
                # Must not contain fragment words
                if any(fw in line_lower for fw in fragment_words):
                    return False
                # Must not contain job title keywords (but allow "Senior" or "Junior" as part of company)
                core_job_keywords = ['manager', 'engineer', 'developer', 'analyst', 'director',
                                    'coordinator', 'specialist', 'consultant', 'architect',
                                    'designer', 'scientist', 'administrator', 'intern']
                if any(kw in line_lower for kw in core_job_keywords):
                    return False
                # Must not contain date patterns
                if re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})', line, re.IGNORECASE):
                    return False
                # Must not look like a location (contains comma or specific location words)
                if location_pattern.match(line):
                    return False
                # All words should be capitalized (proper name) - allow for "&" and minor words
                minor_words = {'and', 'of', 'the', 'for', 'in', 'at', '&'}
                if all(w[0].isupper() or w.lower() in minor_words for w in words if w):
                    return True
            
            return False
        
        def is_job_title_line(line: str) -> bool:
            """Detect if a line is a job title"""
            line_lower = line.lower()
            words = line.split()
            
            # Skip if too long (likely description)
            if len(words) > 8:
                return False
            
            # Check for job title keywords
            title_indicators = job_title_keywords + [
                'head', 'chief', 'principal', 'staff', 'contractor', 'freelancer',
                'full-time', 'part-time', 'contract', 'temporary', 'consultant',
                'assistant', 'deputy', 'vice', 'senior', 'junior', 'entry-level',
                'mid-level', 'team member', 'individual contributor'
            ]
            
            if any(kw in line_lower for kw in title_indicators):
                # Make sure it's not a responsibility (doesn't start with action verb)
                action_starts = ('manage', 'develop', 'lead', 'create', 'implement', 'design')
                if not line_lower.startswith(action_starts):
                    return True
            
            return False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if line starts with a bullet (responsibility)
            if re.match(r'^[•\-\*\▪\▸\→\›\○\●]\s*', line):
                responsibility = re.sub(r'^[•\-\*\▪\▸\→\›\○\●]\s*', '', line)
                if responsibility and current_entry:
                    responsibilities.append(responsibility)
                i += 1
                continue
            
            # Check if line starts with a number (numbered list responsibility)
            if re.match(r'^\d+[\.\)]\s+', line):
                responsibility = re.sub(r'^\d+[\.\)]\s+', '', line)
                if responsibility and current_entry:
                    responsibilities.append(responsibility)
                i += 1
                continue
            
            # Try all date patterns to match Title | Date or Date | Title formats
            matched = False
            for pattern in date_patterns:
                match = pattern.match(line)
                if match:
                    # Save current entry if exists
                    if current_entry:
                        current_entry['responsibilities'] = responsibilities
                        entries.append(current_entry)
                        responsibilities = []
                    
                    groups = match.groups()
                    # Determine which group is title and which are dates based on pattern
                    if len(groups) == 3:
                        # Check if first group is a date (format 5: date first)
                        if re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d)', groups[0], re.IGNORECASE):
                            start_date = groups[0].strip()
                            end_date = groups[1].strip()
                            title = groups[2].strip() if groups[2] else None
                        else:
                            title = groups[0].strip()
                            start_date = groups[1].strip()
                            end_date = groups[2].strip()
                        
                        current_entry = {
                            "company": current_company,
                            "position": title,
                            "start_date": start_date,
                            "end_date": end_date,
                            "duration": None,
                            "location": current_location,
                            "description": None,
                            "responsibilities": []
                        }
                        matched = True
                        break
            
            if matched:
                i += 1
                continue
            
            # Check for standalone date range (sets dates for current/next entry)
            standalone_match = standalone_date_pattern.match(line)
            if standalone_match and current_entry:
                current_entry['start_date'] = standalone_match.group(1).strip()
                current_entry['end_date'] = standalone_match.group(2).strip()
                i += 1
                continue
            
            # Check for company with date on same line
            company_date_match = company_date_pattern.match(line)
            if company_date_match:
                if current_entry:
                    current_entry['responsibilities'] = responsibilities
                    entries.append(current_entry)
                    responsibilities = []
                
                current_company = company_date_match.group(1).strip()
                current_entry = {
                    "company": current_company,
                    "position": None,
                    "start_date": company_date_match.group(2).strip(),
                    "end_date": company_date_match.group(3).strip(),
                    "duration": None,
                    "location": current_location,
                    "description": None,
                    "responsibilities": []
                }
                i += 1
                continue
            
            # Check if this is a job title line (when company is already known)
            if current_company and not current_entry and is_job_title_line(line):
                current_entry = {
                    "company": current_company,
                    "position": line,
                    "start_date": None,
                    "end_date": None,
                    "duration": None,
                    "location": current_location,
                    "description": None,
                    "responsibilities": []
                }
                i += 1
                continue
            
            # Check if this is a company name line
            if is_company_line(line):
                # Save current entry if exists
                if current_entry:
                    current_entry['responsibilities'] = responsibilities
                    entries.append(current_entry)
                    responsibilities = []
                    current_entry = None
                
                # Set new current company
                current_company = line
                current_location = None  # Reset location for new company
                
                # Look ahead for location on next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if location_pattern.match(next_line):
                        current_location = next_line
                
                i += 1
                continue
            
            # Check if this is a location line (sets location for current company context)
            if location_pattern.match(line):
                current_location = line
                i += 1
                continue
            
            # Check for fragment words - these are responsibilities, not entries
            is_fragment = any(fw in line.lower() for fw in fragment_words)
            
            # If not a bullet but starts with capital and looks like responsibility, add it
            if current_entry and is_fragment:
                responsibilities.append(line)
                i += 1
                continue
            
            # Handle lines that might be responsibilities without bullets
            if current_entry and len(line) > 30 and line[0].isupper():
                responsibilities.append(line)
            
            i += 1
        
        # Don't forget the last entry
        if current_entry:
            current_entry['responsibilities'] = responsibilities
            entries.append(current_entry)
        
        return entries
    
    def _extract_skills(self, doc, phrase_matches) -> List[str]:
        """Extract skills from document with categorization"""
        skills = set()
        text = doc.text
        
        # Words/phrases that should not be considered skills
        skip_skill_phrases = {
            'india business development', 'utilize erp systems', 'effective tracking systems',
            'erp systems', 'technology', 'team development', 'strategic business development',
            'india', 'canada', 'usa', 'the', 'and', 'for', 'with', 'from', 'page',
            'experience', 'education', 'summary', 'contact', 'references', 'objective',
            # Company names and other non-skill items
            'aluminum window designs', 'dashmesh metals', 'dineing', 'swiggy',
            'present', 'accuracy', 'efficiency', 'output', 'productivity', 'growth',
            'branding', 'disruptions', 'process', 'schedules',
            # Fragment phrases that shouldn't be skills - common patterns
            'and customer engagement', 'and enhance operational transparency',
            'customer engagement.', 'operational transparency.',
            'and customer engagement.', 'and enhance operational transparency.',
            'recruited and trained staff', 'trained staff', 'recruited and',
            # Date patterns that slip through
            'november 2023', 'december 2023', 'january 2024', 'present',
            'november 2023 – present', 'december 2023 – present',
            # Software names with parentheses (malformed parsing)
            'gainsight)', 'salesforce)', 'hubspot)', '(gainsight', '(salesforce', '(hubspot',
        }
        
        # Job titles that should NOT be considered skills
        job_title_words = {
            'supervisor', 'manager', 'engineer', 'developer', 'analyst', 'lead', 'director',
            'coordinator', 'specialist', 'consultant', 'architect', 'designer', 'scientist',
            'administrator', 'officer', 'executive', 'associate', 'intern', 'trainee',
            'founder', 'co-founder', 'owner', 'partner', 'president', 'vp', 'ceo', 'cto', 'cfo',
            'head', 'chief', 'principal', 'fellow', 'researcher', 'professor', 'lecturer',
            'assistant', 'staff', 'contractor', 'freelancer', 'technician', 'operator',
            'representative', 'agent', 'clerk', 'secretary', 'receptionist', 'team lead',
            'team leader', 'senior', 'junior', 'mid-level', 'entry-level',
            'production associate', 'shipping associate', 'warehouse associate',
            'production & shipping associate', 'production and shipping associate',
        }
        
        # Location names that should NOT be considered skills
        location_words = {
            # Countries
            'india', 'canada', 'usa', 'us', 'uk', 'australia', 'germany', 'france', 'china',
            'japan', 'singapore', 'uae', 'netherlands', 'spain', 'italy', 'brazil', 'mexico',
            # Indian cities/states
            'delhi', 'new delhi', 'mumbai', 'bangalore', 'bengaluru', 'chennai', 'hyderabad',
            'kolkata', 'pune', 'ahmedabad', 'jaipur', 'lucknow', 'chandigarh', 'haryana',
            'hisar', 'gurugram', 'gurgaon', 'noida', 'faridabad', 'karnal', 'ambala',
            'punjab', 'rajasthan', 'gujarat', 'maharashtra', 'karnataka', 'tamil nadu',
            'uttar pradesh', 'west bengal', 'andhra pradesh', 'telangana', 'kerala',
            # Canadian cities/provinces
            'toronto', 'vancouver', 'montreal', 'calgary', 'edmonton', 'ottawa', 'winnipeg',
            'mississauga', 'brampton', 'hamilton', 'kitchener', 'london', 'victoria',
            'woodbridge', 'markham', 'richmond hill', 'scarborough', 'etobicoke', 'north york',
            'ontario', 'quebec', 'british columbia', 'alberta', 'manitoba', 'saskatchewan',
            # US cities/states
            'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
            'san francisco', 'seattle', 'boston', 'austin', 'denver', 'atlanta', 'miami',
            'california', 'texas', 'florida', 'washington', 'massachusetts', 'colorado',
            'georgia', 'illinois', 'pennsylvania', 'ohio', 'michigan', 'arizona',
            # Other common locations
            'remote', 'hybrid', 'on-site', 'onsite',
        }
        
        # Words that indicate this is a sentence/phrase, not a skill
        sentence_indicators = {
            'address', 'oversee', 'ensure', 'provide', 'develop', 'manage', 'create',
            'implement', 'maintain', 'utilize', 'coordinate', 'support', 'facilitate',
            'foster', 'including', 'exceptional', 'various', 'multiple', 'different',
            'proactively', 'effectively', 'successfully', 'and exceed', 'recruited',
            'trained', 'coached', 'mentored', 'supervised', 'led', 'managed',
        }
        
        # Common company suffixes to detect company names in skills
        company_indicators = ['ltd', 'inc', 'corp', 'llc', 'metals', 'designs', 'solutions',
                              'industries', 'enterprises', 'technologies', 'systems', 'services']
        
        # Extract using phrase matcher
        for match_id, start, end in phrase_matches:
            if self.nlp.vocab.strings[match_id] == "SKILLS":
                skill = doc[start:end].text
                # Normalize case
                if skill.lower() not in skip_skill_phrases:
                    skills.add(skill.title() if skill.islower() else skill)
        
        # Try to extract from skills section - expanded section names
        skills_section = self._extract_section(text, [
            'skills', 'technical skills', 'core competencies', 'competencies', 
            'expertise', 'proficiencies', 'technologies', 'tools', 
            'programming languages', 'top skills', 'key skills', 'skill set',
            'professional skills', 'relevant skills', 'areas of expertise',
            'technical proficiencies', 'software skills', 'computer skills',
            'it skills', 'hard skills', 'soft skills', 'transferable skills',
            'technical competencies', 'qualifications', 'abilities',
            'specialized skills', 'industry knowledge'
        ])
        
        if skills_section:
            # Handle multiple skill formats
            
            # Format 1: Comma/bullet/pipe separated list
            skill_candidates = re.split(r'[,•\|;]\s*|\n', skills_section)
            
            # Format 2: Category-based skills (e.g., "Programming: Python, Java, C++")
            category_pattern = re.findall(
                r'([A-Za-z\s]+):\s*([^:\n]+)',
                skills_section
            )
            for category, skill_list in category_pattern:
                for skill in re.split(r'[,;•|]\s*', skill_list):
                    skill = skill.strip()
                    if skill and len(skill) > 1:
                        skill_candidates.append(skill)
            
            # Format 3: Skill with proficiency level (e.g., "Python (Advanced)")
            proficiency_pattern = re.findall(
                r'([A-Za-z\s\+\#\.]+)\s*\([^)]*(?:expert|advanced|intermediate|beginner|proficient|fluent|basic)[^)]*\)',
                skills_section,
                re.IGNORECASE
            )
            skill_candidates.extend(proficiency_pattern)
            
            # Format 4: Skill with years experience (e.g., "Python - 5 years")
            years_pattern = re.findall(
                r'([A-Za-z\s\+\#\.]+)\s*[-–]\s*\d+\s*(?:years?|yrs?)',
                skills_section,
                re.IGNORECASE
            )
            skill_candidates.extend(years_pattern)
            
            # Format 5: Skills with rating bars/dots/stars (common in modern templates)
            rating_pattern = re.findall(
                r'^([A-Za-z\s\+\#\.]+?)(?:\s*[★☆●○◐◑▪▫■□]+|\s*\d+\s*%|\s*\d+/\d+)',
                skills_section,
                re.MULTILINE
            )
            skill_candidates.extend(rating_pattern)
            
            for candidate in skill_candidates:
                candidate = candidate.strip()
                # Clean up common prefixes/suffixes
                candidate = re.sub(r'^[-–—•*]\s*', '', candidate)
                candidate = re.sub(r'\s*[-–—:]\s*$', '', candidate)
                
                if 1 <= len(candidate.split()) <= 4 and 2 <= len(candidate) <= 50:
                    candidate_lower = candidate.lower()
                    # Filter out section headers, generic words, skip phrases, job titles, and locations
                    if (candidate_lower not in self.section_headers and 
                        candidate_lower not in self.skip_words and
                        candidate_lower not in skip_skill_phrases and
                        candidate_lower not in job_title_words and
                        candidate_lower not in location_words):
                        # Also skip if it contains date patterns
                        has_date = bool(re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', candidate_lower))
                        has_year = bool(re.search(r'\b(20\d{2}|19\d{2})\b', candidate))
                        has_present = bool(re.search(r'[-–—]\s*(present|current|now)', candidate_lower))
                        if not has_date and not has_year and not has_present:
                            skills.add(candidate)
        
        # Add common programming language/technology detection
        tech_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|Scala|R|MATLAB)\b',
            r'\b(React|Angular|Vue|Node\.?js|Django|Flask|Spring|Rails|Laravel|Express)\b',
            r'\b(AWS|Azure|GCP|Google Cloud|Kubernetes|Docker|Terraform|Jenkins|Ansible)\b',
            r'\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|Oracle|SQL Server)\b',
            r'\b(TensorFlow|PyTorch|Keras|scikit-learn|Pandas|NumPy|Spark|Hadoop)\b',
            r'\b(Git|GitHub|GitLab|Bitbucket|SVN|Mercurial)\b',
            r'\b(Linux|Unix|Windows|macOS|Ubuntu|CentOS|Debian)\b',
            r'\b(REST|GraphQL|gRPC|WebSocket|OAuth|JWT|SAML)\b',
            r'\b(Agile|Scrum|Kanban|DevOps|CI/CD|TDD|BDD)\b',
            r'\b(HTML5?|CSS3?|SASS|LESS|Bootstrap|Tailwind)\b',
            r'\b(Figma|Sketch|Adobe XD|Photoshop|Illustrator)\b',
            r'\b(Jira|Confluence|Trello|Asana|Slack|Notion)\b',
            r'\b(ERP|SAP|CRM|Salesforce|HubSpot)\b',
            r'\b(Excel|PowerPoint|Word|Outlook|Microsoft Office)\b',
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                skills.add(match)
        
        # Clean and deduplicate
        cleaned_skills = set()
        for skill in skills:
            # Normalize whitespace
            skill = ' '.join(skill.split())
            skill_lower = skill.lower()
            
            # Remove trailing periods and clean up
            skill = skill.rstrip('.,;:')
            skill_lower = skill_lower.rstrip('.,;:')
            
            # Skip invalid skills
            if len(skill) < 2 or len(skill) > 50:
                continue
            if skill_lower in skip_skill_phrases:
                continue
            
            # Skip if it's a job title (exact match)
            if skill_lower in job_title_words:
                continue
            
            # Skip if it CONTAINS a job title word (partial match for phrases like "Production Associate")
            job_title_keywords = {'associate', 'manager', 'engineer', 'developer', 'analyst', 'specialist',
                                  'coordinator', 'supervisor', 'director', 'executive', 'consultant',
                                  'administrator', 'architect', 'lead', 'officer', 'representative',
                                  'assistant', 'intern', 'trainee', 'founder', 'co-founder', 'ceo', 'cto', 'cfo'}
            if any(f' {kw}' in f' {skill_lower} ' or skill_lower.endswith(kw) for kw in job_title_keywords):
                continue
            
            # Skip if it's a location (exact match)
            if skill_lower in location_words:
                continue
            
            # Skip if it contains location indicators (city, state patterns)
            if any(loc in skill_lower for loc in ['india', 'canada', 'ontario', 'haryana', 'delhi', 'mumbai', 
                                                    'bangalore', 'toronto', 'vancouver', 'california', 'new york']):
                continue
            
            # Skip if it looks like a sentence or phrase (contains sentence indicators)
            if any(indicator in skill_lower for indicator in sentence_indicators):
                continue
            
            # Skip if it looks like a company name
            if any(indicator in skill_lower for indicator in company_indicators):
                continue
            
            # Skip if starts with action verb (likely a responsibility, not a skill)
            action_verbs = ('utilize ', 'develop ', 'manage ', 'create ', 'implement ',
                           'provide ', 'ensure ', 'maintain ', 'address ', 'oversee ',
                           'recruited ', 'trained ', 'coached ', 'and ')
            if skill_lower.startswith(action_verbs):
                continue
            
            # Skip if it starts with "And " (sentence fragment)
            if skill_lower.startswith('and '):
                continue
            
            # Skip if it's too long and looks like a phrase (>4 words usually not a skill)
            word_count = len(skill.split())
            if word_count > 4:
                continue
            
            # Skip if it contains punctuation that indicates a sentence (comma, period inside)
            if ',' in skill or '. ' in skill:
                continue
            
            # Skip if it ends with a period (likely incomplete sentence)
            if skill.endswith('.'):
                continue
            
            # Skip if it contains date patterns (Month Year, YYYY, etc.)
            if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', skill_lower):
                continue
            if re.search(r'\b(20\d{2}|19\d{2})\b', skill):
                continue
            if re.search(r'[-–—]\s*(present|current|now)', skill_lower):
                continue
            
            # Skip if it contains ")" without "(" - likely a fragment from (Tool) extraction
            if ')' in skill and '(' not in skill:
                continue
            
            # Skip common non-skill words
            non_skill_words = {'present', 'current', 'remote', 'hybrid', 'full-time', 'part-time',
                              'contract', 'permanent', 'temporary', 'freelance'}
            if skill_lower in non_skill_words:
                continue
            
            # Preserve common acronyms
            if skill.upper() in ['AWS', 'GCP', 'SQL', 'API', 'REST', 'HTML', 'CSS', 'PHP', 'CI/CD', 'TDD', 'BDD', 'ERP', 'CRM', 'SAP', 'HRM']:
                cleaned_skills.add(skill.upper())
            elif skill_lower in ['javascript', 'typescript', 'python', 'java', 'node.js', 'react', 'angular', 'vue', 'salesforce', 'hubspot', 'gainsight']:
                cleaned_skills.add(skill.title())
            else:
                # Title case but preserve known acronyms within
                cleaned_skills.add(skill)
        
        return sorted(list(cleaned_skills))
    
    def _extract_certifications(self, doc) -> List[Dict[str, Any]]:
        """Extract certifications with details"""
        certifications = []
        text = doc.text
        
        cert_keywords = ["certified", "certification", "license", "licensed", "certificate",
                        "credential", "accredited", "accreditation"]
        
        # Words to skip as certification names (too short or generic)
        skip_cert_names = {'ms', 'ca', 'safe', 'aws', 'gcp', 'it', 'ai', 'ml', 'pm', 
                          'hr', 'qa', 'ba', 'sa', 'the', 'and', 'for', 'with', 'from'}
        
        # Common certification patterns - more specific
        cert_patterns = [
            # AWS certifications
            r'(AWS\s+Certified\s+(?:Solutions Architect|Developer|SysOps|DevOps|Cloud Practitioner|'
            r'Data Analytics|Machine Learning|Security|Database|Networking)(?:\s*[-–]\s*(?:Associate|Professional|Specialty))?)',
            # Google certifications
            r'(Google\s+(?:Cloud\s+)?(?:Professional|Associate)\s+(?:Cloud|Data|ML|DevOps)\s+(?:Engineer|Architect|Analyst))',
            # Microsoft certifications
            r'(Microsoft\s+Certified[:\s]+[A-Za-z\s]+(?:Associate|Expert|Fundamentals))',
            r'(Azure\s+(?:Administrator|Developer|Architect|Data|AI|Security|DevOps)\s+(?:Associate|Expert|Fundamentals))',
            # PMP, Scrum, Agile
            r'((?:PMP|Project Management Professional))',
            r'((?:Certified\s+)?Scrum\s+Master(?:\s+[I]{1,3})?)',
            r'((?:PSM|CSM)\s*[I]{0,3})',
            r'(SAFe\s+\d*\s*(?:Agilist|Practitioner|Scrum Master)?)',
            r'(Agile\s+Certified\s+Practitioner)',
            # Cisco
            r'(CCNA|CCNP|CCIE)(?:\s+(?:Routing|Security|Enterprise|Data Center))?',
            r'(Cisco\s+Certified\s+[A-Za-z\s]+)',
            # CompTIA
            r'(CompTIA\s+(?:A\+|Network\+|Security\+|Cloud\+|Linux\+|CySA\+|PenTest\+|CASP\+))',
            # Security certs
            r'(CISSP|CISM|CISA|CEH|OSCP)',
            # Six Sigma
            r'((?:Lean\s+)?Six\s+Sigma\s+(?:Green|Black|Yellow|White)\s+Belt)',
            # Finance certs
            r'(CPA|CFA|CMA|ACCA)(?:\s+Level\s*\d)?',
            r'(Chartered\s+(?:Accountant|Financial\s+Analyst))',
            # Freight/Logistics
            r'(Certified\s+International\s+Freight\s+Forwarder(?:\s*\(CIFF\))?)',
            r'(CIFF)',
            # Generic certification pattern - must be substantial
            r'(Certified\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)',
        ]
        
        # Try to extract from certifications section first
        cert_section = self._extract_section(text, ['certifications', 'certificates', 'licenses',
                                                     'professional certifications', 'credentials'])
        
        if cert_section:
            lines = cert_section.split('\n')
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                    
                # Stop if we hit another section header
                line_lower = line.lower()
                if any(header in line_lower for header in ['projects', 'achievements', 'additional', 
                       'references', 'interests', 'hobbies', 'volunteer', 'awards']):
                    break
                    
                # Clean up bullet points and non-breaking spaces
                line = line.replace('\xa0', ' ')
                line = re.sub(r'^[•\-\*\▪\▸→›○●]\s*', '', line).strip()
                
                # Skip lines that look like descriptions (too long or contain action verbs)
                if len(line) > 80:
                    continue
                action_verbs = ['improved', 'developed', 'created', 'managed', 'led', 'launched',
                               'successfully', 'increased', 'reduced', 'achieved', 'delivered']
                if any(verb in line_lower for verb in action_verbs):
                    continue
                
                # Extract date if present
                date_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{4})', line)
                issue_date = date_match.group(1) if date_match else None
                
                # Extract credential ID if present
                id_match = re.search(r'(?:ID|Credential|License)[:\s#]*([A-Z0-9\-]+)', line, re.IGNORECASE)
                credential_id = id_match.group(1) if id_match else None
                
                # Clean the certification name
                cert_name = line
                if date_match:
                    cert_name = cert_name[:date_match.start()].strip()
                if id_match:
                    cert_name = cert_name[:id_match.start()].strip()
                cert_name = re.sub(r'\s*[-–—:,]\s*$', '', cert_name)
                
                # For items in a certification section, be more lenient
                # Accept anything that starts with a capital and has at least 3 characters
                if len(cert_name) >= 3 and cert_name.lower() not in skip_cert_names:
                    if re.match(r'^[A-Z]', cert_name):
                        # Handle comma-separated items like "Salesforce, HubSpot, Twillio"
                        if ',' in cert_name:
                            items = [item.strip() for item in cert_name.split(',')]
                            for item in items:
                                if len(item) >= 3 and item.lower() not in skip_cert_names:
                                    certifications.append({
                                        "name": item,
                                        "issuing_organization": None,
                                        "issue_date": issue_date,
                                        "expiry_date": None,
                                        "credential_id": credential_id
                                    })
                        else:
                            certifications.append({
                                "name": cert_name,
                                "issuing_organization": None,
                                "issue_date": issue_date,
                                "expiry_date": None,
                                "credential_id": credential_id
                            })
        
        # Use regex patterns for known certifications
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cert_name = match.strip()
                # Skip short/invalid names
                if len(cert_name) < 3 or cert_name.lower() in skip_cert_names:
                    continue
                # Check if already found
                if not any(cert_name.lower() in c.get('name', '').lower() for c in certifications):
                    certifications.append({
                        "name": cert_name,
                        "issuing_organization": None,
                        "issue_date": None,
                        "expiry_date": None,
                        "credential_id": None
                    })
        
        # Deduplicate and clean
        seen = set()
        unique_certs = []
        for cert in certifications:
            name = cert['name']
            name_lower = name.lower()
            
            # Skip very short names or known false positives
            if len(name) < 4 or name_lower in skip_cert_names:
                continue
            
            # Normalize for deduplication
            if name_lower not in seen:
                seen.add(name_lower)
                unique_certs.append(cert)
        
        return unique_certs[:15]  # Limit to 15 certifications
    
    def _extract_projects(self, doc) -> List[Dict[str, Any]]:
        """Extract projects with details"""
        projects = []
        text = doc.text
        
        project_keywords = ["project", "developed", "created", "built", "designed", 
                           "implemented", "architected", "launched", "deployed"]
        
        # Try to extract from projects section
        proj_section = self._extract_section(text, ['projects', 'personal projects', 'side projects',
                                                     'academic projects', 'key projects'])
        
        if proj_section:
            lines = proj_section.split('\n')
            current_project = None
            description_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_project:
                        current_project['description'] = ' '.join(description_lines)
                        projects.append(current_project)
                        current_project = None
                        description_lines = []
                    continue
                
                # Check if this looks like a project title (short, possibly with technologies)
                # Project titles are usually short and might have tech stack after colon/dash
                is_bullet = re.match(r'^[•\-\*\▪\▸→›○●]\s*', line)
                clean_line = re.sub(r'^[•\-\*\▪\▸→›○●]\s*', '', line)
                
                # Check for project title patterns
                title_patterns = [
                    r'^([A-Z][A-Za-z0-9\s\-]+)\s*[-–—:|]\s*(.+)$',  # Title - Technologies
                    r'^([A-Z][A-Za-z0-9\s\-]{3,50})$',  # Short title line
                ]
                
                is_title = False
                for tp in title_patterns:
                    tm = re.match(tp, clean_line)
                    if tm:
                        if current_project:
                            current_project['description'] = ' '.join(description_lines)
                            projects.append(current_project)
                            description_lines = []
                        
                        project_name = tm.group(1).strip()
                        technologies = tm.group(2).strip() if len(tm.groups()) > 1 else None
                        
                        current_project = {
                            "name": project_name,
                            "description": None,
                            "technologies": technologies.split(',') if technologies else [],
                            "url": None,
                            "start_date": None,
                            "end_date": None
                        }
                        is_title = True
                        break
                
                if not is_title and current_project:
                    # This is a description line
                    description_lines.append(clean_line)
                
                # Check for URLs
                url_match = re.search(r'(https?://[^\s]+|github\.com/[^\s]+)', line, re.IGNORECASE)
                if url_match and current_project:
                    url = url_match.group(1)
                    if not url.startswith('http'):
                        url = 'https://' + url
                    current_project['url'] = url
            
            # Don't forget the last project
            if current_project:
                current_project['description'] = ' '.join(description_lines)
                projects.append(current_project)
        
        # Fallback: sentence-based extraction
        if not projects:
            for sent in doc.sents:
                sent_lower = sent.text.lower()
                if any(keyword in sent_lower for keyword in project_keywords):
                    clean_sent = re.sub(r'[^a-zA-Z0-9\s\-\.]', ' ', sent.text).strip()
                    clean_sent = ' '.join(clean_sent.split())
                    if 20 < len(clean_sent) < 500:
                        projects.append({
                            "name": None,
                            "description": clean_sent,
                            "technologies": [],
                            "url": None,
                            "start_date": None,
                            "end_date": None
                        })
        
        return projects[:10]  # Limit to 10 projects
    
    def _extract_summary(self, doc) -> Optional[str]:
        """Extract professional summary or objective"""
        text = doc.text
        
        # Try to extract from summary/objective section
        summary_section = self._extract_section(text, ['summary', 'professional summary', 'profile',
                                                        'professional profile', 'about', 'about me',
                                                        'objective', 'career objective', 'executive summary',
                                                        'personal statement', 'introduction'])
        
        if summary_section:
            # Clean up the section
            lines = [line.strip() for line in summary_section.split('\n') if line.strip()]
            summary = ' '.join(lines)
            # Limit length
            if len(summary) > 1500:
                summary = summary[:1500] + '...'
            return summary if len(summary) > 20 else None
        
        # Fallback: Look for first paragraph that looks like a summary
        lines = text.split('\n')
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            # Skip short lines and section headers
            if len(line) < 50 or any(h in line.lower() for h in self.section_headers):
                continue
            # Skip lines with contact info
            if any(c in line.lower() for c in ['@', 'phone', 'email', 'linkedin', 'github']):
                continue
            # If this looks like a summary paragraph
            if re.search(r'\b(experienced|professional|skilled|passionate|dedicated|motivated|results-driven)\b', line, re.IGNORECASE):
                return line[:1000] if len(line) > 1000 else line
        
        return None
    
    def _extract_languages(self, doc) -> List[Dict[str, str]]:
        """Extract spoken languages with proficiency levels"""
        languages = []
        text = doc.text
        
        # Common languages
        language_names = {
            'english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'chinese',
            'mandarin', 'cantonese', 'japanese', 'korean', 'arabic', 'hindi', 'punjabi',
            'urdu', 'bengali', 'russian', 'dutch', 'swedish', 'norwegian', 'danish',
            'finnish', 'polish', 'turkish', 'greek', 'hebrew', 'persian', 'farsi',
            'thai', 'vietnamese', 'indonesian', 'malay', 'tagalog', 'filipino', 'swahili'
        }
        
        proficiency_levels = {
            'native': 'Native',
            'bilingual': 'Bilingual',
            'fluent': 'Fluent',
            'proficient': 'Proficient',
            'advanced': 'Advanced',
            'intermediate': 'Intermediate',
            'conversational': 'Conversational',
            'basic': 'Basic',
            'beginner': 'Beginner',
            'elementary': 'Elementary',
            'working proficiency': 'Working Proficiency',
            'professional': 'Professional',
            'limited': 'Limited',
            'a1': 'A1 (Beginner)',
            'a2': 'A2 (Elementary)',
            'b1': 'B1 (Intermediate)',
            'b2': 'B2 (Upper Intermediate)',
            'c1': 'C1 (Advanced)',
            'c2': 'C2 (Proficient)'
        }
        
        # Try to extract from languages section
        lang_section = self._extract_section(text, ['languages', 'language skills', 
                                                     'language proficiency', 'spoken languages'])
        
        if lang_section:
            lines = lang_section.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Look for language names in this line
                for lang in language_names:
                    if lang in line_lower:
                        # Find proficiency
                        proficiency = None
                        for prof_key, prof_value in proficiency_levels.items():
                            if prof_key in line_lower:
                                proficiency = prof_value
                                break
                        
                        languages.append({
                            "language": lang.title(),
                            "proficiency": proficiency or "Not specified"
                        })
        
        # Fallback: search entire document for language mentions with proficiency
        if not languages:
            for lang in language_names:
                # Look for patterns like "English (Native)" or "French - Fluent"
                patterns = [
                    rf'\b{lang}\b\s*[\(\[\-–—:]\s*([A-Za-z\s]+)[\)\]]?',
                    rf'\b{lang}\b\s*[-–—:]\s*([A-Za-z\s]+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        prof_text = match.group(1).strip().lower()
                        proficiency = None
                        for prof_key, prof_value in proficiency_levels.items():
                            if prof_key in prof_text:
                                proficiency = prof_value
                                break
                        
                        # Avoid duplicates
                        if not any(l['language'].lower() == lang for l in languages):
                            languages.append({
                                "language": lang.title(),
                                "proficiency": proficiency or prof_text.title()
                            })
        
        return languages
    
    def _extract_achievements(self, doc) -> List[str]:
        """Extract achievements and awards"""
        achievements = []
        text = doc.text
        
        # Try to extract from achievements/awards section
        achieve_section = self._extract_section(text, ['achievements', 'awards', 'honors', 
                                                        'accomplishments', 'recognition',
                                                        'awards and honors', 'achievements and awards'])
        
        if achieve_section:
            lines = achieve_section.split('\n')
            for line in lines:
                line = line.strip()
                if not line or len(line) < 10:
                    continue
                # Clean up bullet points
                line = re.sub(r'^[•\-\*\▪\▸→›○●]\s*', '', line)
                if len(line) > 10:
                    achievements.append(line)
        
        # Also look for achievement-related keywords throughout
        achievement_keywords = ['awarded', 'recognized', 'received', 'achieved', 'won', 
                               'recipient', 'honor', 'distinction', 'medal', 'prize',
                               'promoted', 'exceeded', 'ranked', 'top performer']
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            if any(keyword in sent_lower for keyword in achievement_keywords):
                clean_sent = ' '.join(sent_text.split())
                if 20 < len(clean_sent) < 300 and clean_sent not in achievements:
                    achievements.append(clean_sent)
        
        return achievements[:10]  # Limit to 10 achievements