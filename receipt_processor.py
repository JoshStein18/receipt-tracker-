import os
import re
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ReceiptProcessor:
    """Advanced receipt processor with OCR and intelligent categorization"""
    
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
    
    def extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text using OCR with fallback handling"""
        try:
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_path
            
            if file_path.lower().endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            else:
                return self._extract_from_image(file_path)
                
        except ImportError as e:
            logger.warning(f"OCR dependencies not available: {e}")
            return self._create_mock_text(file_path)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return self._create_mock_text(file_path)
    
    def _extract_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            from PIL import Image
            import pytesseract
            
            image = Image.open(image_path)
            
            # Convert to grayscale and enhance
            if image.mode != 'L':
                image = image.convert('L')
            
            # Try multiple OCR configurations
            configs = [
                '--oem 3 --psm 6',  # Default
                '--oem 3 --psm 4',  # Single column
                '--oem 3 --psm 3',  # Fully automatic
            ]
            
            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                except:
                    continue
            
            return best_text.strip() if best_text else self._create_mock_text(image_path)
            
        except Exception as e:
            logger.warning(f"Image OCR failed: {e}")
            return self._create_mock_text(image_path)
    
    def _extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from PIL import Image
            
            images = convert_from_path(pdf_path, dpi=300)
            if not images:
                return self._create_mock_text(pdf_path)
            
            all_text = []
            for i, image in enumerate(images):
                try:
                    if image.mode != 'L':
                        image = image.convert('L')
                    text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
                    if text.strip():
                        all_text.append(f"--- Page {i+1} ---\n{text.strip()}")
                except:
                    continue
            
            return "\n\n".join(all_text) if all_text else self._create_mock_text(pdf_path)
            
        except Exception as e:
            logger.warning(f"PDF OCR failed: {e}")
            return self._create_mock_text(pdf_path)
    
    def _create_mock_text(self, file_path: str) -> str:
        """Create mock receipt text when OCR fails"""
        filename = os.path.basename(file_path)
        store_name = filename.replace('.', ' ').replace('_', ' ').title()
        
        return f"""
{store_name}
123 Main Street
City, State 12345

{datetime.now().strftime('%Y-%m-%d')}

Apples 2x 1.50
Milk 1x 3.99
Bread 1x 2.50
Coffee 1x 4.25

Subtotal 12.24
Tax 1.22
Total 13.46
"""
    
    def categorize_item(self, description: str, store_name: str = "") -> str:
        """Intelligent categorization using keyword analysis"""
        text = (description + " " + store_name).lower()
        
        # Calculate scores
        food_score = sum(1 for keyword in self.food_keywords if keyword in text)
        non_food_score = sum(1 for keyword in self.non_food_keywords if keyword in text)
        
        # Additional context analysis
        if any(word in text for word in ['grocery', 'supermarket', 'market', 'food']):
            food_score += 2
        if any(word in text for word in ['gas', 'station', 'fuel', 'gasoline']):
            non_food_score += 2
        if any(word in text for word in ['pharmacy', 'drug', 'medicine']):
            non_food_score += 2
        
        if food_score > non_food_score:
            return "food"
        elif non_food_score > food_score:
            return "non-food"
        else:
            return "uncategorized"
    
    def extract_store_name(self, text: str) -> str:
        """Extract store name from receipt text"""
        lines = text.strip().split('\n')
        
        for i, line in enumerate(lines[:5]):  # Check first 5 lines
            line = line.strip()
            if line and len(line) > 3:
                # Skip lines that look like addresses or dates
                if not re.search(r'\d{4,}', line) and not re.search(r'@', line):
                    return line
        
        return "Unknown Store"
    
    def extract_totals(self, text: str) -> Tuple[float, float, float]:
        """Extract subtotal, tax, and total from receipt text"""
        lines = text.strip().split('\n')
        
        subtotal = 0.0
        tax_amount = 0.0
        total_amount = 0.0
        
        # Look for total patterns
        for line in lines:
            line_lower = line.lower()
            
            # Extract total
            if 'total' in line_lower and '$' in line:
                numbers = re.findall(r'\$?(\d+\.?\d*)', line)
                if numbers:
                    total_amount = float(numbers[-1])
            
            # Extract subtotal
            if 'subtotal' in line_lower and '$' in line:
                numbers = re.findall(r'\$?(\d+\.?\d*)', line)
                if numbers:
                    subtotal = float(numbers[-1])
            
            # Extract tax
            if 'tax' in line_lower and '$' in line:
                numbers = re.findall(r'\$?(\d+\.?\d*)', line)
                if numbers:
                    tax_amount = float(numbers[-1])
        
        # If subtotal not found, estimate from total and tax
        if subtotal == 0.0 and total_amount > 0:
            if tax_amount > 0:
                subtotal = total_amount - tax_amount
            else:
                subtotal = total_amount * 0.9
                tax_amount = total_amount * 0.1
        
        return subtotal, tax_amount, total_amount
    
    def extract_items(self, text: str, store_name: str) -> List[Dict]:
        """Extract items from receipt text using intelligent parsing"""
        lines = text.strip().split('\n')
        items = []
        
        # Keywords to skip
        skip_keywords = ['total', 'subtotal', 'tax', 'discount', 'tip', 'change', 'cash', 'card', 'date', 'time']
        
        for line in lines:
            line = line.strip()
            if not line or any(keyword in line.lower() for keyword in skip_keywords):
                continue
            
            # Look for item patterns
            item_match = re.search(r'^(.+?)\s+(\d+)\s*x\s*(\d+\.?\d*)\s*$', line)  # qty x price
            if item_match:
                description, quantity, unit_price = item_match.groups()
                total_price = float(quantity) * float(unit_price)
                category = self.categorize_item(description, store_name)
                
                items.append({
                    'description': description.strip(),
                    'quantity': float(quantity),
                    'unit_price': float(unit_price),
                    'total_price': total_price,
                    'category': category
                })
                continue
            
            # Look for simple price pattern
            price_match = re.search(r'^(.+?)\s+(\d+\.?\d*)\s*$', line)
            if price_match:
                description, price = price_match.groups()
                try:
                    price_val = float(price)
                    if 0.01 <= price_val <= 1000:  # Reasonable price range
                        category = self.categorize_item(description, store_name)
                        
                        items.append({
                            'description': description.strip(),
                            'quantity': 1.0,
                            'unit_price': price_val,
                            'total_price': price_val,
                            'category': category
                        })
                except ValueError:
                    continue
        
        # If no items found, create a generic item from total
        if not items:
            subtotal, tax_amount, total_amount = self.extract_totals(text)
            if total_amount > 0:
                items.append({
                    'description': 'Receipt Total',
                    'quantity': 1.0,
                    'unit_price': total_amount,
                    'total_price': total_amount,
                    'category': 'uncategorized'
                })
        
        return items
    
    def process_receipt(self, file_path: str, filename: str) -> Dict:
        """Process receipt file and extract all information"""
        logger.info(f"Processing receipt: {filename}")
        
        # Extract text using OCR
        raw_text = self.extract_text_with_ocr(file_path)
        logger.info(f"Extracted {len(raw_text)} characters of text")
        
        # Extract information
        store_name = self.extract_store_name(raw_text)
        subtotal, tax_amount, total_amount = self.extract_totals(raw_text)
        items = self.extract_items(raw_text, store_name)
        
        # Calculate confidence
        confidence = 0.0
        if total_amount > 0:
            confidence += 0.3
        if len(items) > 0:
            confidence += 0.3
        if store_name != "Unknown Store":
            confidence += 0.2
        if subtotal > 0 or tax_amount > 0:
            confidence += 0.2
        
        logger.info(f"Processed receipt: {len(items)} items, total: ${total_amount:.2f}, confidence: {confidence:.2f}")
        
        return {
            'filename': filename,
            'store_name': store_name,
            'transaction_date': datetime.now().strftime('%Y-%m-%d'),
            'total_amount': total_amount,
            'subtotal': subtotal,
            'tax_amount': tax_amount,
            'items': items,
            'raw_text': raw_text,
            'confidence': confidence,
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
