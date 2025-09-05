import json
import os
import re
from typing import Dict, List, Optional
import logging

from .models import ReceiptItem, ReceiptData

logger = logging.getLogger(__name__)

class IntelligentReceiptCategorizer:
    """Intelligent receipt categorization using multiple analysis methods"""

    def __init__(self):
        # Food indicators (comprehensive and intelligent)
        self.food_indicators = {
            'direct': ['food', 'eat', 'meal', 'snack', 'drink', 'beverage', 'coffee', 'tea', 'water', 'juice', 'soda'],
            'items': ['pizza', 'burger', 'sandwich', 'salad', 'soup', 'pasta', 'chicken', 'beef', 'fish', 'pork', 'lamb', 'turkey', 'ham', 'bacon', 'sausage', 'cheese', 'milk', 'yogurt', 'butter', 'eggs', 'bread', 'cereal', 'rice', 'noodles', 'apples', 'bananas', 'oranges', 'grapes', 'strawberries', 'vegetables', 'fruits', 'snacks', 'candy', 'chocolate', 'cookies', 'cake', 'pie', 'ice cream', 'frozen', 'fresh', 'organic', 'produce', 'meat', 'dairy', 'seafood', 'poultry', 'vegetarian', 'vegan'],
            'preparation': ['grilled', 'fried', 'baked', 'roasted', 'steamed', 'boiled', 'raw', 'cooked', 'fresh', 'frozen', 'canned', 'dried', 'smoked', 'marinated', 'seasoned'],
            'meals': ['breakfast', 'lunch', 'dinner', 'brunch', 'appetizer', 'entree', 'dessert', 'side', 'combo', 'value', 'meal'],
            'stores': ['restaurant', 'cafe', 'diner', 'bistro', 'grill', 'pizza', 'burger', 'subway', 'mcdonald', 'kfc', 'taco bell', 'chipotle', 'starbucks', 'dunkin', 'grocery', 'supermarket', 'market', 'whole foods', 'safeway', 'kroger', 'walmart', 'target', 'costco', 'trader joe', 'aldi', 'food', 'kitchen', 'deli', 'bakery', 'butcher']
        }
        
        # Non-food indicators
        self.non_food_indicators = {
            'direct': ['gas', 'fuel', 'electric', 'water', 'utility', 'power', 'energy', 'internet', 'phone', 'cable', 'insurance', 'rent', 'mortgage', 'loan', 'payment', 'fee', 'service', 'repair', 'maintenance', 'medicine', 'prescription', 'medical', 'doctor', 'hospital', 'clinic', 'pharmacy'],
            'items': ['gasoline', 'petrol', 'electricity', 'water bill', 'internet bill', 'phone bill', 'cable bill', 'insurance', 'rent', 'mortgage', 'loan payment', 'service fee', 'repair', 'maintenance', 'medicine', 'prescription', 'medical', 'doctor', 'hospital', 'clinic', 'pharmacy', 'computer', 'laptop', 'desktop', 'tablet', 'phone', 'smartphone', 'electronics', 'tech', 'software', 'hardware', 'cable', 'charger', 'battery', 'headphones', 'speaker', 'monitor', 'keyboard', 'mouse', 'webcam', 'microphone', 'camera', 'gaming', 'console', 'controller', 'clothing', 'apparel', 'shoes', 'accessories', 'furniture', 'appliance', 'garden', 'lawn', 'office', 'supplies', 'stationery', 'paper', 'pen', 'pencil', 'staples', 'office depot', 'business', 'work', 'professional', 'home', 'depot', 'lowes', 'hardware', 'furniture', 'appliance', 'garden', 'lawn', 'maintenance', 'repair', 'improvement'],
            'stores': ['gas station', 'shell', 'exxon', 'chevron', 'bp', 'mobil', 'speedway', '7-eleven', 'electric company', 'water company', 'internet provider', 'phone company', 'cable company', 'insurance company', 'bank', 'credit union', 'pharmacy', 'cvs', 'walgreens', 'rite aid', 'computer store', 'best buy', 'microcenter', 'newegg', 'apple store', 'microsoft store', 'dell', 'hp', 'lenovo', 'clothing store', 'shoe store', 'furniture store', 'appliance store', 'garden center', 'home depot', 'lowes', 'office depot', 'staples']
        }

    def get_categories(self) -> Dict:
        """Get available categories"""
        return {"food": {"description": "Food and dining expenses"}, "non-food": {"description": "Non-food expenses"}}

    def categorize_receipt(self, receipt_data: ReceiptData) -> ReceiptData:
        """Intelligently categorize all items in a receipt"""
        logger.info(f"Categorizing receipt from {receipt_data.store_name}")
        
        # Analyze the overall receipt context
        receipt_context = self._analyze_receipt_context(receipt_data)
        logger.info(f"Receipt context: {receipt_context}")
        
        for item in receipt_data.items:
            item.category = self._intelligent_categorize_item(item, receipt_data, receipt_context)
            logger.info(f"Categorized '{item.description}' as {item.category}")
        
        return receipt_data

    def _analyze_receipt_context(self, receipt_data: ReceiptData) -> Dict:
        """Analyze the overall context of the receipt"""
        context = {
            'store_type': 'unknown',
            'food_indicators': 0,
            'non_food_indicators': 0,
            'total_amount': receipt_data.total_amount,
            'item_count': len(receipt_data.items),
            'avg_item_price': 0,
            'price_range': 'unknown'
        }
        
        if receipt_data.items:
            prices = [item.total_price for item in receipt_data.items if item.total_price > 0]
            if prices:
                context['avg_item_price'] = sum(prices) / len(prices)
                min_price, max_price = min(prices), max(prices)
                if max_price - min_price < 5:  # Small price range
                    context['price_range'] = 'uniform'
                elif max_price > min_price * 3:  # Large price range
                    context['price_range'] = 'varied'
        
        # Analyze store name
        store_lower = receipt_data.store_name.lower()
        for indicator_type, indicators in self.food_indicators.items():
            for indicator in indicators:
                if indicator in store_lower:
                    context['food_indicators'] += 1
                    if indicator_type == 'stores':
                        context['store_type'] = 'food'
        
        for indicator_type, indicators in self.non_food_indicators.items():
            for indicator in indicators:
                if indicator in store_lower:
                    context['non_food_indicators'] += 1
                    if indicator_type == 'stores':
                        context['store_type'] = 'non_food'
        
        return context

    def _intelligent_categorize_item(self, item, receipt_data, context: Dict) -> str:
        """Intelligently categorize a single item using multiple analysis methods"""
        description = item.description.lower()
        
        # Method 1: Direct keyword analysis with scoring
        food_score = self._calculate_food_score(description, item, context)
        non_food_score = self._calculate_non_food_score(description, item, context)
        
        # Method 2: Price pattern analysis
        price_analysis = self._analyze_price_patterns(item, context)
        
        # Method 3: Store context analysis
        store_analysis = self._analyze_store_context(receipt_data.store_name, context)
        
        # Method 4: Item description pattern analysis
        pattern_analysis = self._analyze_item_patterns(description, item)
        
        # Method 5: Receipt structure analysis
        structure_analysis = self._analyze_receipt_structure(item, receipt_data, context)
        
        # Combine all analyses with weights
        final_food_score = (food_score * 2.0 + 
                           price_analysis['food'] * 1.5 + 
                           store_analysis['food'] * 2.0 + 
                           pattern_analysis['food'] * 1.0 +
                           structure_analysis['food'] * 1.0)
        
        final_non_food_score = (non_food_score * 2.0 + 
                               price_analysis['non_food'] * 1.5 + 
                               store_analysis['non_food'] * 2.0 + 
                               pattern_analysis['non_food'] * 1.0 +
                               structure_analysis['non_food'] * 1.0)
        
        logger.info(f"Item '{item.description}': food_score={final_food_score:.2f}, non_food_score={final_non_food_score:.2f}")
        
        # Make decision with confidence threshold
        if final_food_score > final_non_food_score and final_food_score > 1.0:
            return "food"
        elif final_non_food_score > final_food_score and final_non_food_score > 1.0:
            return "non-food"
        else:
            # Default based on store type if scores are equal or low
            if context['store_type'] == 'food':
                return "food"
            elif context['store_type'] == 'non_food':
                return "non_food"
            else:
                # Default to food for ambiguous cases (most receipts are food)
                return "food"

    def _calculate_food_score(self, description: str, item, context: Dict) -> float:
        """Calculate food likelihood score"""
        score = 0.0
        
        # Direct food indicators
        for indicator in self.food_indicators['direct']:
            if re.search(r'\b' + re.escape(indicator) + r'\b', description):
                score += 2.0
        
        # Food items
        for indicator in self.food_indicators['items']:
            if re.search(r'\b' + re.escape(indicator) + r'\b', description):
                score += 3.0
        
        # Preparation methods
        for indicator in self.food_indicators['preparation']:
            if re.search(r'\b' + re.escape(indicator) + r'\b', description):
                score += 1.5
        
        # Meal types
        for indicator in self.food_indicators['meals']:
            if re.search(r'\b' + re.escape(indicator) + r'\b', description):
                score += 2.5
        
        return score

    def _calculate_non_food_score(self, description: str, item, context: Dict) -> float:
        """Calculate non-food likelihood score"""
        score = 0.0
        
        # Direct non-food indicators
        for indicator in self.non_food_indicators['direct']:
            if re.search(r'\b' + re.escape(indicator) + r'\b', description):
                score += 2.0
        
        # Non-food items
        for indicator in self.non_food_indicators['items']:
            if re.search(r'\b' + re.escape(indicator) + r'\b', description):
                score += 3.0
        
        return score

    def _analyze_price_patterns(self, item, context: Dict) -> Dict:
        """Analyze price patterns to determine category"""
        food_score = 0.0
        non_food_score = 0.0
        
        price = item.total_price
        
        # Food items typically have different price ranges
        if 0.50 <= price <= 50.00:  # Common food item range
            food_score += 1.0
        elif price > 100.00:  # Expensive items are often non-food
            non_food_score += 1.0
        
        # Compare to average item price
        if context['avg_item_price'] > 0:
            if price <= context['avg_item_price'] * 1.5:  # Similar to other items
                food_score += 0.5
            else:  # Much more expensive than average
                non_food_score += 0.5
        
        # Analyze price patterns
        if context['price_range'] == 'uniform' and 1.0 <= price <= 20.0:
            food_score += 1.0  # Uniform prices often indicate food items
        elif context['price_range'] == 'varied' and price > 50.0:
            non_food_score += 1.0  # Varied high prices often indicate non-food
        
        return {'food': food_score, 'non_food': non_food_score}

    def _analyze_store_context(self, store_name: str, context: Dict) -> Dict:
        """Analyze store context for categorization hints"""
        food_score = 0.0
        non_food_score = 0.0
        
        store_lower = store_name.lower()
        
        # Strong store indicators
        if any(store in store_lower for store in self.food_indicators['stores']):
            food_score += 3.0
        elif any(store in store_lower for store in self.non_food_indicators['stores']):
            non_food_score += 3.0
        
        # Context from overall receipt
        if context['food_indicators'] > context['non_food_indicators']:
            food_score += 1.5
        elif context['non_food_indicators'] > context['food_indicators']:
            non_food_score += 1.5
        
        return {'food': food_score, 'non_food': non_food_score}

    def _analyze_item_patterns(self, description: str, item) -> Dict:
        """Analyze item description patterns"""
        food_score = 0.0
        non_food_score = 0.0
        
        # Look for quantity patterns (food often has quantities)
        if re.search(r'\d+\s*x\s*\d+', description):  # "2 x 3.50" pattern
            food_score += 1.0
        
        # Look for size indicators (food often has sizes)
        if re.search(r'\b(small|medium|large|regular|big|small|mini|jumbo|tall|grande|venti)\b', description):
            food_score += 1.5
        
        # Look for brand names (non-food often has specific brands)
        if re.search(r'\b(apple|samsung|microsoft|dell|hp|lenovo|nike|adidas|sony|lg|canon|nikon)\b', description):
            non_food_score += 2.0
        
        # Look for model numbers (non-food often has model numbers)
        if re.search(r'\b[A-Z]{2,}\d+\b', description):  # Pattern like "ABC123"
            non_food_score += 1.5
        
        # Look for food-specific patterns
        if re.search(r'\b(oz|lb|pound|gallon|liter|kg|gram)\b', description):
            food_score += 1.0
        
        # Look for non-food specific patterns
        if re.search(r'\b(model|version|edition|gb|tb|mhz|ghz|inch|inch)\b', description):
            non_food_score += 1.5
        
        return {'food': food_score, 'non_food': non_food_score}

    def _analyze_receipt_structure(self, item, receipt_data, context: Dict) -> Dict:
        """Analyze receipt structure for categorization hints"""
        food_score = 0.0
        non_food_score = 0.0
        
        # If this is the only item and it's expensive, likely non-food
        if len(receipt_data.items) == 1 and item.total_price > 50:
            non_food_score += 2.0
        
        # If there are many items with similar prices, likely food
        if len(receipt_data.items) > 3:
            similar_prices = 0
            for other_item in receipt_data.items:
                if other_item != item and abs(other_item.total_price - item.total_price) < 5:
                    similar_prices += 1
            if similar_prices > 1:
                food_score += 1.0
        
        # If item appears multiple times, likely food
        item_count = sum(1 for i in receipt_data.items if i.description.lower() == item.description.lower())
        if item_count > 1:
            food_score += 1.5
        
        return {'food': food_score, 'non_food': non_food_score}
