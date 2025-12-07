import os
import re
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
import pytesseract
from PIL import Image
from typing import Optional
from pathlib import Path

class FileProcessor:
    def __init__(self):
        # Configure Tesseract path (update for your system)
        self.tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        try:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        except:
            pass  # Tesseract might not be available

    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from magic bytes if extension is missing"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
            
            # PDF: starts with %PDF
            if header[:4] == b'%PDF':
                return '.pdf'
            
            # DOCX/ZIP: starts with PK (ZIP archive)
            if header[:2] == b'PK':
                return '.docx'
            
            # PNG: starts with \x89PNG
            if header[:4] == b'\x89PNG':
                return '.png'
            
            # JPEG: starts with \xff\xd8\xff
            if header[:3] == b'\xff\xd8\xff':
                return '.jpg'
            
            # Default to txt if we can't detect
            return '.txt'
        except:
            return '.txt'

    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        
        # If no extension or empty, try to detect from content
        if not file_ext or file_ext == '.':
            file_ext = self._detect_file_type(file_path)
        
        if file_ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_ext == '.docx':
            return self._extract_from_docx(file_path)
        elif file_ext in ('.png', '.jpg', '.jpeg'):
            return self._extract_from_image(file_path)
        elif file_ext == '.txt':
            return self._extract_from_txt(file_path)
        elif file_ext == '.doc':
            # Try to handle .doc as .docx (may not always work)
            try:
                return self._extract_from_docx(file_path)
            except:
                raise ValueError(f"Old .doc format not supported. Please convert to .docx or .pdf")
        else:
            # Try PDF as default (most common resume format)
            try:
                return self._extract_from_pdf(file_path)
            except:
                raise ValueError(f"Unsupported file format: {file_ext}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            text = pdf_extract_text(file_path)
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"PDF extraction failed: {str(e)}")

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            raise ValueError(f"DOCX extraction failed: {str(e)}")

    def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image files using OCR"""
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"Image OCR failed: {str(e)}")

    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Text file reading failed: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing excessive whitespace and special characters"""
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Replace multiple spaces (not newlines) with single space
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Replace 3+ consecutive newlines with 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove form feed characters
        text = re.sub(r'\x0c', '', text)
        # Remove non-printable characters (except newlines, tabs)
        text = ''.join(char for char in text if char.isprintable() or char in {'\n', '\t'})
        return text.strip()