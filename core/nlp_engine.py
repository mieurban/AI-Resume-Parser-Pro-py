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
        
        # Date patterns for experience/education
        date_patterns = [
            # Month Year - Month Year or Present
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "-"}, {"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "-"}, {"LOWER": "present"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "–"}, {"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "Xxxx"}, {"SHAPE": "dddd"}, {"ORTH": "–"}, {"LOWER": "present"}],
            # MM/YYYY - MM/YYYY
            [{"SHAPE": "dd/dddd"}, {"ORTH": "-"}, {"SHAPE": "dd/dddd"}],
            [{"SHAPE": "dd/dddd"}, {"ORTH": "-"}, {"LOWER": "present"}],
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
        """Extract candidate name from document"""
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
                                  'junior', 'intern', 'associate'}:
                    return False
            # All words should be primarily alphabetic and title case or all caps
            for word in words:
                clean_word = word.replace('.', '').replace(',', '').replace("'", "")
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
            suffixes = {'jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'phd', 'md', 'esq'}
            while words and words[-1].lower().strip('.') in suffixes:
                words = words[:-1]
            if len(words) >= 2:
                return ' '.join(word.title() if not word.isupper() else word.title() for word in words)
            return name.title()
        
        text = doc.text
        
        # Strategy 1: Look for explicit name labels
        name_label_patterns = [
            r'(?:name|full\s*name|candidate\s*name)\s*[:\-]\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+)\s*\n',  # Name at start of doc followed by newline
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
        # LinkedIn usernames can contain letters, numbers, hyphens
        # PDF extraction sometimes adds spaces, so we need to handle that
        linkedin_patterns = [
            # Pattern with possible spaces (from PDF extraction issues)
            r'linkedin\.com/in/([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)*)',
            r'www\.linkedin\.com/in/([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)*)',
            r'https?://(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)*)',
        ]
        for pattern in linkedin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Clean up the username - remove spaces and trailing special chars
                username = match.group(1)
                # Remove spaces that shouldn't be there (PDF extraction artifact)
                username = re.sub(r'\s+', '', username)
                # Remove any trailing hyphens or special chars
                username = username.rstrip('-_/')
                # LinkedIn usernames are typically at least 5 chars
                if len(username) >= 5:
                    contact["linkedin"] = f"https://linkedin.com/in/{username}"
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
                             'provide', 'develop', 'ensure', 'implement', 'maintain']
                if any(sw in inst_lower for sw in skip_words):
                    continue
                # Skip if too short
                if len(institution) < 5:
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
        """Parse education entries from a section - handles LinkedIn multi-line format"""
        entries = []
        
        # LinkedIn format patterns
        # Date range pattern: (September 2022 - April 2023) or (2018 - 2021)
        date_range_pattern = r'\(?\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?\s*\d{4})\s*[-–]\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?\s*\d{4}|Present|Current)\s*\)?'
        single_date_pattern = r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}|\d{4})'
        gpa_pattern = r'(?:GPA|CGPA|Grade|Score)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?'
        
        # LinkedIn format: "Degree, Field of Study · (Date Range)"
        linkedin_degree_pattern = r'^([^,·\n]+),\s*([^·\n]+?)(?:\s*·\s*|\s+)\((.+?)\)\s*$'
        
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
        
        for line in normalized_lines:
            # Skip section header
            if line.lower().strip() in ['education', 'education:', 'academic background']:
                continue
            
            # Check if this is an institution line (contains institution keyword, no degree indicators)
            line_lower = line.lower()
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
            
            # Check for standalone date line
            date_range_match = re.search(date_range_pattern, line, re.IGNORECASE)
            if date_range_match:
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
            
            # Fallback: Check for degree keywords in line
            for abbr, full_name in degree_mapping.items():
                if abbr in line_lower or full_name.lower() in line_lower:
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
        """Extract a section from the resume text"""
        # Build regex pattern for section headers
        section_pattern = '|'.join(re.escape(name) for name in section_names)
        
        # Common section headers that might follow
        all_sections = ['education', 'experience', 'skills', 'projects', 'certifications',
                       'achievements', 'awards', 'publications', 'languages', 'interests',
                       'references', 'summary', 'objective', 'profile', 'work history',
                       'employment', 'professional', 'technical', 'personal', 'contact']
        end_sections = [s for s in all_sections if s not in section_names]
        end_pattern = '|'.join(re.escape(name) for name in end_sections)
        
        # Find section start
        start_match = re.search(
            rf'(?:^|\n)\s*({section_pattern})\s*[:\n]',
            text,
            re.IGNORECASE | re.MULTILINE
        )
        
        if not start_match:
            return None
        
        start_pos = start_match.end()
        
        # Find section end
        end_match = re.search(
            rf'(?:^|\n)\s*({end_pattern})\s*[:\n]',
            text[start_pos:],
            re.IGNORECASE | re.MULTILINE
        )
        
        if end_match:
            return text[start_pos:start_pos + end_match.start()]
        else:
            # Take rest of document or limit to reasonable length
            return text[start_pos:start_pos + 3000]
    
    def _extract_experience(self, doc, matches, phrase_matches) -> List[Dict[str, Any]]:
        """Extract work experience with comprehensive details"""
        experience = []
        text = doc.text
        
        # Skip words that are not actual employers
        skip_companies = {'linkedin', 'github', 'twitter', 'facebook', 'instagram', 'youtube',
                          'experience', 'education', 'skills', 'summary', 'team', 'page',
                          'projects', 'certifications', 'achievements', 'references', 'languages',
                          'and', 'the', 'with', 'from', 'for', 'dedicated', 'committed',
                          'experienced', 'professional', 'proven', 'track', 'record'}
        
        # Common company suffixes
        company_suffixes = ['ltd', 'ltd.', 'inc', 'inc.', 'corp', 'corp.', 'llc', 'llp',
                           'company', 'co', 'co.', 'corporation', 'enterprises', 'solutions',
                           'technologies', 'tech', 'systems', 'services', 'group', 'partners',
                           'consulting', 'labs', 'studio', 'studios', 'agency', 'firm',
                           'designs', 'metals', 'industries', 'pvt', 'private', 'limited']
        
        # Job title keywords (standalone titles)
        job_title_keywords = [
            'supervisor', 'manager', 'engineer', 'developer', 'analyst', 'lead', 'director',
            'coordinator', 'specialist', 'consultant', 'architect', 'designer', 'scientist',
            'administrator', 'officer', 'executive', 'associate', 'intern', 'trainee',
            'founder', 'co-founder', 'owner', 'partner', 'president', 'vp', 'ceo', 'cto', 'cfo'
        ]
        
        # Duration pattern (X years Y months)
        duration_pattern = r'(\d+\s*years?\s*\d*\s*months?)'
        
        # Date range patterns
        date_range_patterns = [
            # Month Year - Month Year or Present
            r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–—]\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|[Pp]resent)',
        ]
        
        # Try LinkedIn-style parsing first
        # LinkedIn PDFs often have: "Company Name\n Duration\n Title\n Date Range\n Location"
        linkedin_entries = self._parse_linkedin_experience(text, company_suffixes, job_title_keywords, 
                                                            duration_pattern, date_range_patterns, skip_companies)
        if linkedin_entries:
            experience.extend(linkedin_entries)
        
        # If no entries found, try section-based extraction
        if not experience:
            exp_section = self._extract_section(text, ['experience', 'work experience', 'employment history',
                                                        'work history', 'professional experience',
                                                        'career history', 'employment'])
            
            if exp_section:
                entries = self._parse_experience_section_v2(exp_section, company_suffixes, 
                                                             job_title_keywords, skip_companies)
                experience.extend(entries)
        
        # Use spaCy ORG entities as fallback for companies
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
        for exp in experience:
            company = (exp.get('company') or '').strip()
            position = (exp.get('position') or '').strip()
            
            # Skip if company name is too short or looks like a fragment
            if company:
                company_lower = company.lower()
                if len(company) < 4:
                    continue
                if any(company_lower.startswith(skip) for skip in skip_companies):
                    continue
                # Skip fragmented text
                if company_lower.endswith(('co', 'with', 'from', 'and', 'the', 'a')):
                    continue
            
            key = (company.lower() if company else '', position.lower() if position else '')
            if key != ('', '') and key not in seen:
                seen.add(key)
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
            filtered.append(resp)
        
        # Limit to top 8 responsibilities per entry
        return filtered[:8]
    
    def _parse_linkedin_experience(self, text: str, company_suffixes: List[str], 
                                    job_title_keywords: List[str], duration_pattern: str,
                                    date_range_patterns: List[str], skip_companies: set) -> List[Dict[str, Any]]:
        """Parse LinkedIn-style resume format"""
        entries = []
        
        # Extended skip patterns for fragments
        extended_skip = skip_companies | {
            'utilize', 'provide', 'develop', 'ensure', 'implement', 'maintain',
            'facilitating', 'fostering', 'managing', 'coordinating', 'supporting',
            'systems', 'solutions', 'timely', 'resolutions', 'feedback'
        }
        
        # LinkedIn format often has company name followed by duration on same/next line
        # Pattern: "Company Name Ltd\n2 years 3 months\nJob Title\nMonth Year - Present"
        
        # Find all company names with suffixes - more strict pattern
        company_pattern = r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:' + '|'.join(re.escape(s) for s in company_suffixes) + r'))\b\s*\n?\s*(' + duration_pattern + r')?'
        
        for match in re.finditer(company_pattern, text, re.IGNORECASE):
            company = match.group(1).strip()
            duration = match.group(2).strip() if match.group(2) else None
            
            # Clean company name - remove "Experience " prefix if present
            if company.lower().startswith('experience '):
                company = company[11:].strip()
            
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
        """Parse experience entries from a section - improved version"""
        entries = []
        lines = [l.strip() for l in section.split('\n') if l.strip()]
        
        current_entry = None
        responsibilities = []
        
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
            
            # Check if this line looks like a company name
            has_company_suffix = any(suffix in line.lower() for suffix in company_suffixes)
            
            # Skip lines that contain fragment phrases
            fragment_words = ['utilize', 'provide', 'develop', 'ensure', 'implement', 
                             'maintain', 'facilitating', 'fostering', 'managing']
            is_fragment = any(fw in line.lower() for fw in fragment_words)
            
            # Check if line contains a job title keyword
            has_job_title = any(kw in line.lower() for kw in job_title_keywords)
            
            # Check for date pattern
            has_date = bool(re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}', line, re.IGNORECASE))
            
            # Check for duration pattern
            has_duration = bool(re.search(r'\d+\s*years?\s*\d*\s*months?', line, re.IGNORECASE))
            
            # If we found a new company (not a fragment), save current and start new entry
            if has_company_suffix and len(line) > 5 and not is_fragment:
                if current_entry:
                    current_entry['responsibilities'] = responsibilities
                    entries.append(current_entry)
                    responsibilities = []
                
                # Extract company name
                company = line
                for suffix in company_suffixes:
                    pattern = rf'([A-Z][A-Za-z\s&,\.\-]+{re.escape(suffix)})'
                    cm = re.search(pattern, line, re.IGNORECASE)
                    if cm:
                        company = cm.group(1).strip()
                        break
                
                current_entry = {
                    "company": company,
                    "position": None,
                    "start_date": None,
                    "end_date": None,
                    "duration": None,
                    "location": None,
                    "description": None,
                    "responsibilities": []
                }
            elif current_entry:
                # Update current entry with additional info
                if has_job_title and not current_entry.get('position'):
                    # Extract job title - clean it properly
                    for kw in job_title_keywords:
                        if kw in line.lower():
                            # Get the line as the title (clean it)
                            title = re.sub(r'\d+\s*years?\s*\d*\s*months?', '', line).strip()
                            # Remove date patterns from title
                            title = re.sub(r'\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{0,4}.*$', '', title, flags=re.IGNORECASE).strip()
                            # Also clean standalone month names at end
                            title = re.sub(r'\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)$', '', title, flags=re.IGNORECASE).strip()
                            if 3 < len(title) < 50:
                                current_entry['position'] = title
                            break
                
                if has_date and not current_entry.get('start_date'):
                    date_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\s*[-–—]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|[Pp]resent)', line, re.IGNORECASE)
                    if date_match:
                        current_entry['start_date'] = date_match.group(1)
                        current_entry['end_date'] = date_match.group(2)
                
                if has_duration and not current_entry.get('duration'):
                    dur_match = re.search(r'(\d+\s*years?\s*\d*\s*months?)', line, re.IGNORECASE)
                    if dur_match:
                        current_entry['duration'] = dur_match.group(1)
            
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
            'experience', 'education', 'summary', 'contact', 'references', 'objective'
        }
        
        # Extract using phrase matcher
        for match_id, start, end in phrase_matches:
            if self.nlp.vocab.strings[match_id] == "SKILLS":
                skill = doc[start:end].text
                # Normalize case
                if skill.lower() not in skip_skill_phrases:
                    skills.add(skill.title() if skill.islower() else skill)
        
        # Try to extract from skills section
        skills_section = self._extract_section(text, ['skills', 'technical skills', 'core competencies',
                                                       'competencies', 'expertise', 'proficiencies',
                                                       'technologies', 'tools', 'programming languages',
                                                       'top skills'])
        
        if skills_section:
            # Extract skills from comma/bullet separated list
            skill_candidates = re.split(r'[,•\|;]\s*|\n', skills_section)
            for candidate in skill_candidates:
                candidate = candidate.strip()
                # Clean up common prefixes/suffixes
                candidate = re.sub(r'^[-–—•*]\s*', '', candidate)
                candidate = re.sub(r'\s*[-–—:]\s*$', '', candidate)
                
                if 1 <= len(candidate.split()) <= 4 and 2 <= len(candidate) <= 50:
                    candidate_lower = candidate.lower()
                    # Filter out section headers, generic words, and skip phrases
                    if (candidate_lower not in self.section_headers and 
                        candidate_lower not in self.skip_words and
                        candidate_lower not in skip_skill_phrases):
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
            
            # Skip invalid skills
            if len(skill) < 2 or len(skill) > 50:
                continue
            if skill_lower in skip_skill_phrases:
                continue
            # Skip if starts with action verb (likely a responsibility, not a skill)
            if skill_lower.startswith(('utilize ', 'develop ', 'manage ', 'create ', 'implement ')):
                continue
            
            # Preserve common acronyms
            if skill.upper() in ['AWS', 'GCP', 'SQL', 'API', 'REST', 'HTML', 'CSS', 'PHP', 'CI/CD', 'TDD', 'BDD', 'ERP', 'CRM', 'SAP']:
                cleaned_skills.add(skill.upper())
            elif skill_lower in ['javascript', 'typescript', 'python', 'java', 'node.js', 'react', 'angular', 'vue']:
                cleaned_skills.add(skill.title())
            else:
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
                if not line or len(line) < 8:  # Increased minimum length
                    continue
                # Clean up bullet points
                line = re.sub(r'^[•\-\*\▪\▸→›○●]\s*', '', line)
                
                # Skip lines that are just short words
                if len(line.split()) < 2:
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
                
                # Validate certification name
                if len(cert_name) >= 8 and cert_name.lower() not in skip_cert_names:
                    # Must contain at least one certification-related word or be in proper format
                    is_valid = (
                        any(kw in cert_name.lower() for kw in ['certified', 'certificate', 'certification', 'license']) or
                        any(kw in cert_name.upper() for kw in ['CIFF', 'PMP', 'AWS', 'CCNA', 'CPA', 'CFA']) or
                        re.match(r'^[A-Z]', cert_name)  # Starts with capital
                    )
                    if is_valid:
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