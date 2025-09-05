import os
import uuid
from datetime import datetime
from typing import Optional
import re

def generate_id() -> str:
    """Generate a unique ID for receipts and transactions"""
    return str(uuid.uuid4())

def get_upload_path(data_dir: str, filename: str) -> str:
    """Generate organized upload path with year/month structure"""
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    
    upload_dir = os.path.join(data_dir, "uploads", year, month)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Add timestamp to filename to avoid conflicts
    name, ext = os.path.splitext(filename)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{name}_{timestamp}{ext}"
    
    return os.path.join(upload_dir, safe_filename)

def clean_text(text: str) -> str:
    """Clean and normalize OCR text"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common OCR artifacts
    text = re.sub(r'[^\w\s\.\,\-\$\%\(\)]', '', text)
    
    return text

def extract_currency(text: str) -> Optional[float]:
    """Extract currency amount from text"""
    if not text:
        return None
    
    # Look for currency patterns
    patterns = [
        r'\$(\d+\.?\d*)',  # $123.45
        r'(\d+\.?\d*)\s*dollars?',  # 123.45 dollars
        r'(\d+\.?\d*)\s*USD',  # 123.45 USD
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                continue
    
    return None

def parse_date(text: str) -> Optional[datetime]:
    """Parse date from text using common formats"""
    if not text:
        return None
    
    # Common date patterns
    patterns = [
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                if len(matches[0]) == 3:
                    if pattern.startswith(r'(\d{4})'):  # YYYY-MM-DD
                        year, month, day = matches[0]
                    else:  # MM/DD/YYYY or MM-DD-YYYY
                        month, day, year = matches[0]
                    
                    return datetime(int(year), int(month), int(day))
            except ValueError:
                continue
    
    return None

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing invalid characters"""
    # Remove or replace invalid characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    safe = re.sub(r'_+', '_', safe)
    # Remove leading/trailing underscores
    safe = safe.strip('_')
    
    return safe if safe else "receipt"
