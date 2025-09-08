import os
import re
import json
import logging
import pickle
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import uuid
import base64
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

class SmartMLProcessor:
    """Smart ML processor that can handle any receipt format"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.model_file = 'smart_ml_model.pkl'
        self.vectorizer_file = 'smart_vectorizer.pkl'
        
        # Load existing model if available
        self.load_model()
        
        # If no model exists, train with diverse data
        if not self.is_trained:
            self.train_with_diverse_data()
    
    def create_diverse_training_data(self):
        """Create training data from multiple receipt types and formats"""
        
        # Training data: (text, category)
        training_data = [
            # Food items from various stores
            ("GROUND BEEF", "food"),
            ("GRND BEEF", "food"),
            ("GROUND BEEF 93% LEAN", "food"),
            ("ORGANIC GROUND BEEF", "food"),
            ("BEEF", "food"),
            ("CHICKEN BREAST", "food"),
            ("CHICKEN THIGHS", "food"),
            ("SALMON FILLET", "food"),
            ("TILAPIA", "food"),
            ("PORK CHOPS", "food"),
            ("TURKEY BREAST", "food"),
            ("HAM", "food"),
            ("BACON", "food"),
            ("SAUSAGE", "food"),
            ("DELI MEAT", "food"),
            ("DELI HAM", "food"),
            ("DELI TURKEY", "food"),
            ("DELI CHICKEN", "food"),
            ("MILK", "food"),
            ("WHOLE MILK", "food"),
            ("2% MILK", "food"),
            ("SKIM MILK", "food"),
            ("ORGANIC MILK", "food"),
            ("CHEESE", "food"),
            ("CHEDDAR CHEESE", "food"),
            ("MOZZARELLA", "food"),
            ("SWISS CHEESE", "food"),
            ("YOGURT", "food"),
            ("GREEK YOGURT", "food"),
            ("BUTTER", "food"),
            ("EGGS", "food"),
            ("ORGANIC EGGS", "food"),
            ("BREAD", "food"),
            ("WHITE BREAD", "food"),
            ("WHEAT BREAD", "food"),
            ("SOURDOUGH", "food"),
            ("BAGELS", "food"),
            ("CROISSANTS", "food"),
            ("APPLES", "food"),
            ("BANANAS", "food"),
            ("ORANGES", "food"),
            ("GRAPES", "food"),
            ("STRAWBERRIES", "food"),
            ("BLUEBERRIES", "food"),
            ("VEGETABLES", "food"),
            ("BROCCOLI", "food"),
            ("CARROTS", "food"),
            ("LETTUCE", "food"),
            ("SPINACH", "food"),
            ("TOMATOES", "food"),
            ("ONIONS", "food"),
            ("POTATOES", "food"),
            ("RICE", "food"),
            ("BROWN RICE", "food"),
            ("PASTA", "food"),
            ("SPAGHETTI", "food"),
            ("CEREAL", "food"),
            ("OATMEAL", "food"),
            ("COFFEE", "food"),
            ("TEA", "food"),
            ("JUICE", "food"),
            ("ORANGE JUICE", "food"),
            ("APPLE JUICE", "food"),
            ("SODA", "food"),
            ("COKE", "food"),
            ("PEPSI", "food"),
            ("WATER", "food"),
            ("BEER", "food"),
            ("WINE", "food"),
            ("SNACKS", "food"),
            ("CHIPS", "food"),
            ("CRACKERS", "food"),
            ("NUTS", "food"),
            ("ALMONDS", "food"),
            ("WALNUTS", "food"),
            ("COOKIES", "food"),
            ("CANDY", "food"),
            ("CHOCOLATE", "food"),
            ("ICE CREAM", "food"),
            ("FROZEN VEGETABLES", "food"),
            ("FROZEN PIZZA", "food"),
            ("FROZEN DINNER", "food"),
            ("BAKERY", "food"),
            ("CAKE", "food"),
            ("PIE", "food"),
            ("MUFFINS", "food"),
            ("DONUTS", "food"),
            ("CONDIMENTS", "food"),
            ("KETCHUP", "food"),
            ("MUSTARD", "food"),
            ("MAYO", "food"),
            ("SALAD DRESSING", "food"),
            ("SPICES", "food"),
            ("SALT", "food"),
            ("PEPPER", "food"),
            ("GARLIC", "food"),
            ("CANNED GOODS", "food"),
            ("SOUP", "food"),
            ("BEANS", "food"),
            ("TOMATO SAUCE", "food"),
            ("BABY FOOD", "food"),
            ("FORMULA", "food"),
            ("PET FOOD", "food"),
            ("DOG FOOD", "food"),
            ("CAT FOOD", "food"),
            
            # Non-food items
            ("GAS", "non-food"),
            ("FUEL", "non-food"),
            ("GASOLINE", "non-food"),
            ("GAS STATION", "non-food"),
            ("FUEL PUMP", "non-food"),
            ("ELECTRIC", "non-food"),
            ("WATER BILL", "non-food"),
            ("UTILITY", "non-food"),
            ("POWER", "non-food"),
            ("ENERGY", "non-food"),
            ("INTERNET", "non-food"),
            ("PHONE", "non-food"),
            ("CELL PHONE", "non-food"),
            ("UBER", "non-food"),
            ("LYFT", "non-food"),
            ("TAXI", "non-food"),
            ("BUS", "non-food"),
            ("TRAIN", "non-food"),
            ("PARKING", "non-food"),
            ("PHARMACY", "non-food"),
            ("DRUG", "non-food"),
            ("MEDICINE", "non-food"),
            ("MEDICAL", "non-food"),
            ("PRESCRIPTION", "non-food"),
            ("MOVIE", "non-food"),
            ("CINEMA", "non-food"),
            ("THEATER", "non-food"),
            ("NETFLIX", "non-food"),
            ("SPOTIFY", "non-food"),
            ("ENTERTAINMENT", "non-food"),
            ("GYM", "non-food"),
            ("FITNESS", "non-food"),
            ("CLOTHING", "non-food"),
            ("SHOES", "non-food"),
            ("ELECTRONICS", "non-food"),
            ("COMPUTER", "non-food"),
            ("LAPTOP", "non-food"),
            ("PHONE", "non-food"),
            ("TABLET", "non-food"),
            ("SOFTWARE", "non-food"),
            ("HARDWARE", "non-food"),
            ("CABLE", "non-food"),
            ("CHARGER", "non-food"),
            ("BATTERY", "non-food"),
            ("HEADPHONES", "non-food"),
            ("MONITOR", "non-food"),
            ("KEYBOARD", "non-food"),
            ("MOUSE", "non-food"),
            ("CAMERA", "non-food"),
            ("GAMING", "non-food"),
            ("OFFICE SUPPLIES", "non-food"),
            ("PAPER", "non-food"),
            ("PEN", "non-food"),
            ("PENCIL", "non-food"),
            ("FURNITURE", "non-food"),
            ("APPLIANCE", "non-food"),
            ("HOME GOODS", "non-food"),
            ("TOYS", "non-food"),
            ("GAMES", "non-food"),
            ("BOOKS", "non-food"),
            ("MAGAZINES", "non-food"),
            ("BEAUTY", "non-food"),
            ("COSMETICS", "non-food"),
            ("PERSONAL CARE", "non-food"),
            ("AUTOMOTIVE", "non-food"),
            ("CAR", "non-food"),
            ("TIRES", "non-food"),
            ("OIL", "non-food"),
            ("PARTS", "non-food"),
            ("GARDEN", "non-food"),
            ("OUTDOOR", "non-food"),
            ("TRAVEL", "non-food"),
            ("LUGGAGE", "non-food"),
            ("HOTEL", "non-food"),
            ("SPORTS", "non-food"),
            ("ATHLETIC", "non-food"),
            ("BABY", "non-food"),
            ("KIDS", "non-food"),
            ("CHILDREN", "non-food"),
            ("PET", "non-food"),
            ("ANIMAL", "non-food"),
            ("VETERINARY", "non-food"),
        ]
        
        return training_data
    
    def train_with_diverse_data(self):
        """Train the ML model with diverse receipt data"""
        logger.info("Training ML model with diverse receipt data...")
        
        # Get training data
        training_data = self.create_diverse_training_data()
        
        # Separate text and labels
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Fit and transform the text data
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Smart ML model trained successfully! Accuracy: {accuracy:.2f}")
        logger.info(f"Training samples: {len(texts)}")
        logger.info(f"Features: {X.shape[1]}")
        
        # Save model
        self.save_model()
        self.is_trained = True
        
        return accuracy
    
    def save_model(self):
        """Save trained model and vectorizer"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info("Smart model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load existing trained model and vectorizer"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.vectorizer_file):
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.is_trained = True
                logger.info("Smart model loaded successfully")
            else:
                logger.info("No existing smart model found, will train new one")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_trained = False
    
    def categorize_with_ml(self, text: str) -> str:
        """Use trained ML model to categorize text"""
        if not self.is_trained:
            logger.warning("Model not trained, using fallback categorization")
            return self._fallback_categorize(text)
        
        try:
            # Transform text using trained vectorizer
            X = self.vectorizer.transform([text])
            
            # Predict category
            prediction = self.model.predict(X)[0]
            confidence = self.model.predict_proba(X)[0].max()
            
            logger.info(f"Smart ML categorization: '{text}' -> {prediction} (confidence: {confidence:.2f})")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Smart ML categorization failed: {e}")
            return self._fallback_categorize(text)
    
    def _fallback_categorize(self, text: str) -> str:
        """Fallback categorization using keywords"""
        text_lower = text.lower()
        
        food_keywords = ['food', 'eat', 'meal', 'snack', 'drink', 'beverage', 'coffee', 'tea',
                        'breakfast', 'lunch', 'dinner', 'cafe', 'restaurant', 'dining',
                        'pizza', 'burger', 'sandwich', 'salad', 'soup', 'pasta', 'chicken',
                        'beef', 'fish', 'vegetarian', 'vegan', 'organic', 'fresh', 'produce',
                        'apples', 'bananas', 'oranges', 'grapes', 'strawberries', 'milk',
                        'cheese', 'yogurt', 'butter', 'eggs', 'bread', 'cereal', 'rice',
                        'grocery', 'supermarket', 'market', 'dairy', 'fruit', 'vegetable']
        
        non_food_keywords = ['gas', 'fuel', 'gasoline', 'station', 'shell', 'exxon', 'chevron',
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
                            'apple', 'samsung', 'microsoft', 'google', 'best buy', 'dell', 'hp']
        
        food_score = sum(1 for keyword in food_keywords if keyword in text_lower)
        non_food_score = sum(1 for keyword in non_food_keywords if keyword in text_lower)
        
        if food_score > non_food_score:
            return "food"
        elif non_food_score > food_score:
            return "non-food"
        else:
            return "uncategorized"
    
    def extract_text_with_google_vision(self, file_path: str) -> str:
        """Extract text using Google Vision API"""
        try:
            from google.cloud import vision
            import io
            
            # Initialize the client
            client = vision.ImageAnnotatorClient()
            
            # Read the image file
            with io.open(file_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Perform text detection
            response = client.text_detection(image=image)
            texts = response.text_annotations
            
            if texts:
                # Return the first (full) text annotation
                return texts[0].description
            else:
                logger.warning("No text found in image")
                return ""
                
        except ImportError:
            logger.warning("Google Vision API not available")
            return ""
        except Exception as e:
            logger.warning(f"Google Vision API failed: {e}")
            return ""
    
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
            if 'total' in line_lower:
                numbers = re.findall(r'\$?(\d+\.?\d*)', line)
                if numbers:
                    total_amount = float(numbers[-1])
            
            # Extract subtotal
            if 'subtotal' in line_lower:
                numbers = re.findall(r'\$?(\d+\.?\d*)', line)
                if numbers:
                    subtotal = float(numbers[-1])
            
            # Extract tax
            if 'tax' in line_lower:
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
    
    def smart_extract_items(self, text: str, store_name: str) -> List[Dict]:
        """Smart extraction of items from any receipt format"""
        lines = text.strip().split('\n')
        items = []
        processed_lines = set()
        
        # Keywords to skip
        skip_keywords = ['total', 'subtotal', 'tax', 'discount', 'tip', 'change', 'cash', 'card', 'date', 'time', 'payment', 'visa', 'auth', 'return', 'survey', 'help', 'password', 'user id', 'thank', 'visit', 'receipt', 'receipt #', 'transaction', 'authorization', 'phone', 'address', 'street', 'city', 'state', 'zip', 'pm', 'am']
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or any(keyword in line.lower() for keyword in skip_keywords):
                continue
            
            # Skip if we've already processed this line
            if i in processed_lines:
                continue
            
            # Skip lines that look like addresses or phone numbers
            if re.search(r'\d{3}-\d{3}-\d{4}', line) or re.search(r'\d{3}\s+\d{3}\s+\d{4}', line):
                continue
            
            # Skip lines that look like addresses
            if re.search(r'\d+\s+[A-Za-z\s]+(?:St|Ave|Rd|Blvd|Dr|Ln|Way|Pl|Ct)', line):
                continue
            
            # Look for various item patterns
            item_found = False
            
            # Pattern 1: "ITEM DESCRIPTION 2 @ $7.99 ea"
            qty_price_match = re.search(r'^(.+?)\s+(\d+)\s*@\s*\$?(\d+\.?\d*)\s*ea', line)
            if qty_price_match:
                description, quantity, unit_price = qty_price_match.groups()
                total_price = float(quantity) * float(unit_price)
                category = self.categorize_with_ml(description)
                
                items.append({
                    'description': description.strip(),
                    'quantity': float(quantity),
                    'unit_price': float(unit_price),
                    'total_price': total_price,
                    'category': category
                })
                processed_lines.add(i)
                item_found = True
                continue
            
            # Pattern 2: "ITEM DESCRIPTION 2 x 7.99"
            item_match = re.search(r'^(.+?)\s+(\d+)\s*x\s*(\d+\.?\d*)\s*$', line)
            if item_match:
                description, quantity, unit_price = item_match.groups()
                total_price = float(quantity) * float(unit_price)
                category = self.categorize_with_ml(description)
                
                items.append({
                    'description': description.strip(),
                    'quantity': float(quantity),
                    'unit_price': float(unit_price),
                    'total_price': total_price,
                    'category': category
                })
                processed_lines.add(i)
                item_found = True
                continue
            
            # Pattern 3: "ITEM DESCRIPTION $12.99" or "ITEM DESCRIPTION 12.99"
            price_match = re.search(r'^(.+?)\s+\$?(\d+\.?\d*)\s*$', line)
            if price_match:
                description, price = price_match.groups()
                try:
                    price_val = float(price)
                    if 0.01 <= price_val <= 1000:  # Reasonable price range
                        # Make sure description is not just a number or address
                        if not re.match(r'^\d+$', description.strip()) and len(description.strip()) > 2:
                            category = self.categorize_with_ml(description)
                            
                            items.append({
                                'description': description.strip(),
                                'quantity': 1.0,
                                'unit_price': price_val,
                                'total_price': price_val,
                                'category': category
                            })
                            processed_lines.add(i)
                            item_found = True
                except ValueError:
                    pass
            
            # Pattern 4: Multi-line items (description on one line, price on next)
            if not item_found and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Check if current line looks like description and next line looks like price
                if (len(line) > 5 and not re.search(r'\$?\d+\.?\d*', line) and 
                    re.search(r'\$?\d+\.?\d*', next_line) and 
                    not any(keyword in next_line.lower() for keyword in skip_keywords) and
                    not any(keyword in line.lower() for keyword in skip_keywords)):
                    
                    price_match = re.search(r'\$?(\d+\.?\d*)', next_line)
                    if price_match:
                        price_val = float(price_match.group(1))
                        if 0.01 <= price_val <= 1000:
                            category = self.categorize_with_ml(line)
                            
                            items.append({
                                'description': line.strip(),
                                'quantity': 1.0,
                                'unit_price': price_val,
                                'total_price': price_val,
                                'category': category
                            })
                            processed_lines.add(i)
                            processed_lines.add(i + 1)
                            item_found = True
        
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
        """Process receipt file using smart ML model"""
        logger.info(f"Processing receipt with smart ML model: {filename}")
        
        # Extract text using Google Vision API
        raw_text = self.extract_text_with_google_vision(file_path)
        
        if not raw_text:
            logger.warning("No text extracted from image, using fallback")
            return self._create_fallback_data(filename)
        
        logger.info(f"Text extracted: {len(raw_text)} characters")
        
        # Extract information
        store_name = self.extract_store_name(raw_text)
        subtotal, tax_amount, total_amount = self.extract_totals(raw_text)
        items = self.smart_extract_items(raw_text, store_name)
        
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
        
        logger.info(f"Smart ML processing complete: {len(items)} items, total: ${total_amount:.2f}, confidence: {confidence:.2f}")
        
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
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ml_provider': 'Smart ML Model (Any Receipt Format)'
        }
    
    def _create_fallback_data(self, filename: str) -> Dict:
        """Create fallback data when no text can be extracted"""
        return {
            'filename': filename,
            'store_name': 'Unknown Store',
            'transaction_date': datetime.now().strftime('%Y-%m-%d'),
            'total_amount': 0.0,
            'subtotal': 0.0,
            'tax_amount': 0.0,
            'items': [],
            'raw_text': 'No text extracted from image',
            'confidence': 0.0,
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ml_provider': 'Fallback (No OCR Available)'
        }
