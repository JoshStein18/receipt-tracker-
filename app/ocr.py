import os
import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles OCR processing for images and PDFs"""
    
    def __init__(self):
        # Configure tesseract path if needed (for Railway deployment)
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Resize if too small (minimum 300px width)
            if image.width < 300:
                ratio = 300 / image.width
                new_size = (300, int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image file using OCR"""
        try:
            # Open and preprocess image
            image = Image.open(image_path)
            processed_image = self.preprocess_image(image)
            
            # Try multiple OCR configurations for better results
            configs = [
                r'--oem 3 --psm 6',  # Default
                r'--oem 3 --psm 4',  # Single column
                r'--oem 3 --psm 3',  # Fully automatic
                r'--oem 3 --psm 1',  # Automatic page segmentation with OSD
            ]
            
            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_image, config=config)
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                        logger.info(f"Better OCR result with config: {config}")
                except Exception as e:
                    logger.warning(f"OCR config {config} failed: {e}")
                    continue
            
            text = best_text
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed for image {image_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            if not images:
                logger.warning(f"No pages found in PDF: {pdf_path}")
                return ""
            
            # Extract text from each page
            all_text = []
            for i, image in enumerate(images):
                try:
                    processed_image = self.preprocess_image(image)
                    
                    # Try multiple OCR configurations for PDF pages too
                    configs = [
                        r'--oem 3 --psm 6',  # Default
                        r'--oem 3 --psm 4',  # Single column
                        r'--oem 3 --psm 3',  # Fully automatic
                    ]
                    
                    best_page_text = ""
                    for config in configs:
                        try:
                            page_text = pytesseract.image_to_string(processed_image, config=config)
                            if len(page_text.strip()) > len(best_page_text.strip()):
                                best_page_text = page_text
                        except Exception as e:
                            logger.warning(f"PDF page {i+1} OCR config {config} failed: {e}")
                            continue
                    
                    if best_page_text.strip():
                        all_text.append(f"--- Page {i+1} ---\n{best_page_text.strip()}")
                except Exception as e:
                    logger.warning(f"Failed to process PDF page {i+1}: {e}")
                    continue
            
            return "\n\n".join(all_text)
        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from file (image or PDF)"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
        
        # Determine file type
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in ['.pdf']:
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            return self.extract_text_from_image(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
    
    def get_confidence_score(self, text: str) -> float:
        """Calculate confidence score based on text quality"""
        if not text:
            return 0.0
        
        # Simple confidence scoring based on text characteristics
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        # Check for common receipt elements
        confidence_factors = 0
        total_checks = 0
        
        # Check for currency symbols
        has_currency = any('$' in line for line in non_empty_lines)
        if has_currency:
            confidence_factors += 1
        total_checks += 1
        
        # Check for numbers (likely prices)
        has_numbers = any(re.search(r'\d+\.?\d*', line) for line in non_empty_lines)
        if has_numbers:
            confidence_factors += 1
        total_checks += 1
        
        # Check for reasonable line count (receipts typically have 5-50 lines)
        line_count = len(non_empty_lines)
        if 5 <= line_count <= 50:
            confidence_factors += 1
        total_checks += 1
        
        # Check for common receipt words
        receipt_keywords = ['total', 'subtotal', 'tax', 'receipt', 'thank', 'change', 'cash', 'card']
        has_keywords = any(any(keyword in line.lower() for keyword in receipt_keywords) for line in non_empty_lines)
        if has_keywords:
            confidence_factors += 1
        total_checks += 1
        
        return confidence_factors / total_checks if total_checks > 0 else 0.0
