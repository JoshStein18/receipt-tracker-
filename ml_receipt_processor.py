import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import uuid

import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class MLReceiptProcessor:
    """ML-powered receipt processing with intelligent item extraction and categorization"""
    
    def __init__(self):
        self.food_keywords = [
            'food', 'eat', 'meal', 'snack', 'drink', 'beverage', 'coffee', 'tea',
            'breakfast', 'lunch', 'dinner', 'cafe', 'restaurant', 'dining',
            'pizza', 'burger', 'sandwich', 'salad', 'soup', 'pasta', 'chicken',
            'beef', 'fish', 'vegetarian', 'vegan', 'organic', 'fresh', 'produce',
            'apples', 'bananas', 'oranges', 'grapes', 'strawberries', 'milk',
            'cheese', 'yogurt', 'butter', 'eggs', 'bread', 'cereal', 'rice',
            'grocery', 'supermarket', 'market', 'dairy', 'fruit', 'vegetable',
            'walmart', 'safeway', 'kroger', 'whole foods', 'costco',
            'mcdonald', 'burger king', 'subway', 'starbucks', 'dunkin',
            'chipotle', 'taco bell', 'kfc', 'pizza hut', 'domino'
        ]
        
        self.non_food_keywords = [
            'gas', 'fuel', 'gasoline', 'station', 'shell', 'exxon', 'chevron',
            'electric', 'water', 'utility', 'power', 'energy', 'internet', 'phone',
            'uber', 'lyft', 'taxi', 'bus', 'train', 'metro', 'transit', 'parking',
            'pharmacy', 'drug', 'medicine', 'medical', 'doctor', 'hospital',
            'cvs', 'walgreens', 'rite aid', 'health', 'prescription',
            'movie', 'cinema', 'theater', 'netflix', 'spotify', 'amazon prime',
            'entertainment', 'sports', 'gym', 'fitness', 'club',
            'clothing', 'apparel', 'shoes', 'accessories', 'electronics', 'amazon',
            'office', 'supplies', 'stationery', 'paper', 'pen', 'pencil', 'staples',
            'home', 'depot', 'lowes', 'hardware', 'furniture', 'appliance',
            'computer', 'laptop', 'desktop', 'tablet', 'phone', 'smartphone', 'tech',
            'software', 'hardware', 'cable', 'charger', 'battery', 'headphones',
            'monitor', 'keyboard', 'mouse', 'webcam', 'camera', 'gaming',
            'apple', 'samsung', 'microsoft', 'google', 'best buy', 'dell', 'hp'
        ]
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Training data for ML categorization
        self.training_data = self._create_training_data()
        self._train_ml_model()
    
    def _create_training_data(self) -> List[Tuple[str, str]]:
        """Create training data for ML categorization"""
        training_data = []
        
        # Food examples
        food_examples = [
            "apple", "banana", "milk", "bread", "cheese", "chicken", "beef", "fish",
            "pizza", "burger", "sandwich", "salad", "soup", "pasta", "rice", "cereal",
            "coffee", "tea", "juice", "soda", "water", "beer", "wine", "snack",
            "candy", "chocolate", "cookie", "cake", "ice cream", "yogurt", "butter",
            "eggs", "vegetables", "fruits", "organic", "fresh", "frozen", "dairy"
        ]
        
        for example in food_examples:
            training_data.append((example, "food"))
        
        # Non-food examples
        non_food_examples = [
            "gas", "fuel", "electricity", "water bill", "internet", "phone", "cable",
            "uber", "lyft", "taxi", "bus", "train", "parking", "toll", "ticket",
            "medicine", "prescription", "pharmacy", "doctor", "hospital", "clinic",
            "movie", "cinema", "theater", "netflix", "spotify", "subscription",
            "clothing", "shirt", "pants", "shoes", "jacket", "dress", "accessories",
            "electronics", "computer", "laptop", "phone", "tablet", "camera",
            "furniture", "chair", "table", "bed", "sofa", "lamp", "appliance",
            "tools", "hardware", "paint", "lumber", "garden", "lawn", "maintenance"
        ]
        
        for example in non_food_examples:
            training_data.append((example, "non-food"))
        
        return training_data
    
    def _train_ml_model(self):
        """Train the ML model for categorization"""
        try:
            texts, labels = zip(*self.training_data)
            self.X_train = self.vectorizer.fit_transform(texts)
            self.labels = list(labels)
            logger.info(f"Trained ML model with {len(self.training_data)} examples")
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            self.X_train = None
            self.labels = []
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR with preprocessing"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Resize for better OCR
            width, height = image.size
            if width < 1000:
                scale = 1000 / width
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Extract text with multiple configurations
            configs = [
                '--oem 3 --psm 6',  # Default
                '--oem 3 --psm 4',  # Single column
                '--oem 3 --psm 3',  # Fully automatic
                '--oem 3 --psm 1',  # Automatic page segmentation with OSD
            ]
            
            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                        logger.info(f"Better OCR result with config: {config}")
                except Exception as e:
                    logger.warning(f"OCR config {config} failed: {e}")
                    continue
            
            return best_text.strip()
            
        except Exception as e:
            logger.error(f"OCR failed for image {image_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            if not images:
                logger.warning(f"No pages found in PDF: {pdf_path}")
                return ""
            
            all_text = []
            for i, image in enumerate(images):
                try:
                    # Convert to grayscale and enhance
                    if image.mode != 'L':
                        image = image.convert('L')
                    
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(2.0)
                    
                    # Extract text
                    text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
                    if text.strip():
                        all_text.append(f"--- Page {i+1} ---\n{text.strip()}")
                except Exception as e:
                    logger.warning(f"Failed to process PDF page {i+1}: {e}")
                    continue
            
            return "\n\n".join(all_text)
            
        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from image or PDF file"""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        else:
            return self.extract_text_from_image(file_path)
    
    def ml_categorize_item(self, description: str) -> str:
        """Use ML to categorize an item"""
        if not self.X_train is not None:
            return self._keyword_categorize(description)
        
        try:
            # Vectorize the description
            desc_vector = self.vectorizer.transform([description])
            
            # Calculate similarity with training data
            similarities = cosine_similarity(desc_vector, self.X_train)[0]
            
            # Get the most similar training example
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity > 0.1:  # Threshold for confidence
                predicted_label = self.labels[best_match_idx]
                logger.info(f"ML categorized '{description}' as '{predicted_label}' (similarity: {best_similarity:.3f})")
                return predicted_label
            else:
                # Fallback to keyword-based categorization
                return self._keyword_categorize(description)
                
        except Exception as e:
            logger.error(f"ML categorization failed for '{description}': {e}")
            return self._keyword_categorize(description)
    
    def _keyword_categorize(self, description: str) -> str:
        """Fallback keyword-based categorization"""
        text = description.lower()
        
        food_score = sum(1 for keyword in self.food_keywords if keyword in text)
        non_food_score = sum(1 for keyword in self.non_food_keywords if keyword in text)
        
        if food_score > non_food_score:
            return "food"
        elif non_food_score > food_score:
            return "non-food"
        else:
            return "uncategorized"
    
    def extract_items_ml(self, text: str) -> List[Dict]:
        """Extract items from receipt text using ML and pattern recognition"""
        lines = text.strip().split('\n')
        items = []
        
        # Patterns for item extraction
        item_patterns = [
            r'^(.+?)\s+(\d+)\s*x\s*(\d+\.?\d*)\s*$',  # Description qty x price
            r'^(.+?)\s+(\d+\.?\d*)\s*$',  # Description followed by price
        ]
        
        # Keywords to skip
        skip_keywords = ['total', 'subtotal', 'tax', 'discount', 'tip', 'change', 'cash', 'card']
        
        for line in lines:
            line = line.strip()
            if not line or any(keyword in line.lower() for keyword in skip_keywords):
                continue
            
            # Try to extract item information
            for pattern in item_patterns:
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        description = groups[0].strip()
                        if len(groups) == 3:  # qty x price format
                            quantity = float(groups[1])
                            unit_price = float(groups[2])
                            total_price = quantity * unit_price
                        else:  # description price format
                            quantity = 1.0
                            unit_price = float(groups[1])
                            total_price = unit_price
                        
                        # Use ML to categorize
                        category = self.ml_categorize_item(description)
                        
                        item = {
                            'description': description,
                            'quantity': quantity,
                            'unit_price': unit_price,
                            'total_price': total_price,
                            'category': category
                        }
                        
                        items.append(item)
                        logger.info(f"Extracted item: {description} - ${total_price:.2f} - {category}")
                        break
        
        # If no items found, try to extract from any line with currency
        if not items:
            logger.info("No items found with patterns, trying currency extraction...")
            for line in lines:
                if '$' in line or re.search(r'\d+\.\d{2}', line):
                    # Extract price
                    price_match = re.search(r'\$?(\d+\.?\d*)', line)
                    if price_match:
                        price = float(price_match.group(1))
                        if 0.01 <= price <= 1000:  # Reasonable price range
                            description = line.replace(price_match.group(0), '').strip()
                            if description:
                                category = self.ml_categorize_item(description)
                                items.append({
                                    'description': description,
                                    'quantity': 1.0,
                                    'unit_price': price,
                                    'total_price': price,
                                    'category': category
                                })
                                logger.info(f"Extracted item from currency: {description} - ${price:.2f} - {category}")
        
        return items
    
    def extract_totals_ml(self, text: str) -> Tuple[float, float, float]:
        """Extract subtotal, tax, and total using ML and pattern recognition"""
        lines = text.strip().split('\n')
        
        subtotal = 0.0
        tax_amount = 0.0
        total_amount = 0.0
        
        # Look for total patterns
        total_patterns = [
            r'total[:\s]*\$?(\d+\.?\d*)',
            r'amount[:\s]*\$?(\d+\.?\d*)',
            r'grand[:\s]*total[:\s]*\$?(\d+\.?\d*)',
        ]
        
        subtotal_patterns = [
            r'subtotal[:\s]*\$?(\d+\.?\d*)',
            r'sub[:\s]*total[:\s]*\$?(\d+\.?\d*)',
        ]
        
        tax_patterns = [
            r'tax[:\s]*\$?(\d+\.?\d*)',
            r'gst[:\s]*\$?(\d+\.?\d*)',
            r'hst[:\s]*\$?(\d+\.?\d*)',
            r'vat[:\s]*\$?(\d+\.?\d*)',
        ]
        
        for line in lines:
            line_lower = line.lower()
            
            # Extract total
            for pattern in total_patterns:
                match = re.search(pattern, line_lower)
                if match:
                    total_amount = float(match.group(1))
                    break
            
            # Extract subtotal
            for pattern in subtotal_patterns:
                match = re.search(pattern, line_lower)
                if match:
                    subtotal = float(match.group(1))
                    break
            
            # Extract tax
            for pattern in tax_patterns:
                match = re.search(pattern, line_lower)
                if match:
                    tax_amount = float(match.group(1))
                    break
        
        # If subtotal not found, estimate from total and tax
        if subtotal == 0.0 and total_amount > 0:
            if tax_amount > 0:
                subtotal = total_amount - tax_amount
            else:
                subtotal = total_amount * 0.9  # Estimate
                tax_amount = total_amount * 0.1  # Estimate
        
        return subtotal, tax_amount, total_amount
    
    def extract_store_name_ml(self, text: str) -> str:
        """Extract store name using ML and pattern recognition"""
        lines = text.strip().split('\n')
        
        # Store name is usually in the first few lines
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if line and len(line) > 3:
                # Skip lines that look like addresses or dates
                if not re.search(r'\d{4,}', line) and not re.search(r'@', line):
                    return line
        
        return "Unknown Store"
    
    def process_receipt_ml(self, file_path: str, filename: str) -> Dict:
        """Process receipt using ML for text extraction and categorization"""
        logger.info(f"Processing receipt with ML: {filename}")
        
        # Extract text
        raw_text = self.extract_text(file_path)
        logger.info(f"Extracted {len(raw_text)} characters of text")
        
        if not raw_text:
            logger.warning("No text extracted from receipt")
            return {
                'filename': filename,
                'store_name': 'Unknown Store',
                'transaction_date': None,
                'total_amount': 0.0,
                'subtotal': 0.0,
                'tax_amount': 0.0,
                'items': [],
                'raw_text': raw_text,
                'confidence': 0.0
            }
        
        # Extract information using ML
        store_name = self.extract_store_name_ml(raw_text)
        subtotal, tax_amount, total_amount = self.extract_totals_ml(raw_text)
        items = self.extract_items_ml(raw_text)
        
        # Calculate confidence based on extracted data
        confidence = 0.0
        if total_amount > 0:
            confidence += 0.3
        if len(items) > 0:
            confidence += 0.3
        if store_name != "Unknown Store":
            confidence += 0.2
        if subtotal > 0 or tax_amount > 0:
            confidence += 0.2
        
        logger.info(f"ML processing complete: {len(items)} items, total: ${total_amount:.2f}, confidence: {confidence:.2f}")
        
        return {
            'filename': filename,
            'store_name': store_name,
            'transaction_date': datetime.now().strftime('%Y-%m-%d'),
            'total_amount': total_amount,
            'subtotal': subtotal,
            'tax_amount': tax_amount,
            'items': items,
            'raw_text': raw_text,
            'confidence': confidence
        }
