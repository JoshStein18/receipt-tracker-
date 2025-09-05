import json
import os
import re
from typing import Dict, List, Optional
import logging

from .models import ReceiptItem, ReceiptData

logger = logging.getLogger(__name__)

class ReceiptCategorizer:
    """Categorizes receipt items based on keyword rules"""
    
    def __init__(self, rules_file: str = None):
        self.rules_file = rules_file or os.path.join(os.path.dirname(__file__), "rules.default.json")
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict:
        """Load categorization rules from JSON file"""
        try:
            if os.path.exists(self.rules_file):
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default rules if file doesn't exist
                default_rules = self._get_default_rules()
                self._save_rules(default_rules)
                return default_rules
        except Exception as e:
            logger.error(f"Failed to load categorization rules: {e}")
            return self._get_default_rules()
    
    def _get_default_rules(self) -> Dict:
        """Get default categorization rules"""
        return {
            "categories": {
                "groceries": {
                    "keywords": [
                        "grocery", "food", "supermarket", "market", "fresh", "produce",
                        "meat", "dairy", "bread", "milk", "eggs", "cheese", "fruit",
                        "vegetable", "organic", "whole foods", "safeway", "kroger",
                        "walmart", "target", "costco", "sam's club"
                    ],
                    "description": "Grocery and food items"
                },
                "restaurants": {
                    "keywords": [
                        "restaurant", "cafe", "coffee", "pizza", "burger", "sandwich",
                        "dining", "food", "eat", "lunch", "dinner", "breakfast",
                        "mcdonald's", "burger king", "subway", "starbucks", "dunkin",
                        "chipotle", "taco bell", "kfc", "pizza hut", "domino's"
                    ],
                    "description": "Restaurant and dining expenses"
                },
                "gas": {
                    "keywords": [
                        "gas", "fuel", "gasoline", "petrol", "station", "shell",
                        "exxon", "chevron", "bp", "mobil", "speedway", "7-eleven"
                    ],
                    "description": "Gas and fuel expenses"
                },
                "utilities": {
                    "keywords": [
                        "electric", "water", "gas", "utility", "power", "energy",
                        "internet", "phone", "cable", "internet", "telecom"
                    ],
                    "description": "Utility bills and services"
                },
                "transportation": {
                    "keywords": [
                        "uber", "lyft", "taxi", "bus", "train", "metro", "transit",
                        "parking", "toll", "highway", "airport", "flight", "rental"
                    ],
                    "description": "Transportation and travel"
                },
                "healthcare": {
                    "keywords": [
                        "pharmacy", "drug", "medicine", "medical", "doctor", "hospital",
                        "clinic", "cvs", "walgreens", "rite aid", "health", "prescription"
                    ],
                    "description": "Healthcare and medical expenses"
                },
                "entertainment": {
                    "keywords": [
                        "movie", "cinema", "theater", "netflix", "spotify", "amazon prime",
                        "game", "entertainment", "sports", "gym", "fitness", "club"
                    ],
                    "description": "Entertainment and recreation"
                },
                "shopping": {
                    "keywords": [
                        "clothing", "apparel", "shoes", "accessories", "electronics",
                        "amazon", "ebay", "online", "store", "retail", "fashion"
                    ],
                    "description": "General shopping and retail"
                },
                "office": {
                    "keywords": [
                        "office", "supplies", "stationery", "paper", "pen", "pencil",
                        "staples", "office depot", "business", "work", "professional"
                    ],
                    "description": "Office supplies and business expenses"
                },
                "home": {
                    "keywords": [
                        "home", "depot", "lowes", "hardware", "furniture", "appliance",
                        "garden", "lawn", "maintenance", "repair", "improvement"
                    ],
                    "description": "Home improvement and maintenance"
                }
            },
            "store_mappings": {
                "walmart": "groceries",
                "target": "shopping",
                "costco": "groceries",
                "safeway": "groceries",
                "kroger": "groceries",
                "whole foods": "groceries",
                "starbucks": "restaurants",
                "mcdonald's": "restaurants",
                "subway": "restaurants",
                "shell": "gas",
                "exxon": "gas",
                "chevron": "gas",
                "cvs": "healthcare",
                "walgreens": "healthcare",
                "amazon": "shopping",
                "home depot": "home",
                "lowes": "home"
            }
        }
    
    def _save_rules(self, rules: Dict) -> None:
        """Save rules to JSON file"""
        try:
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save categorization rules: {e}")
    
    def categorize_receipt(self, receipt_data: ReceiptData) -> ReceiptData:
        """Categorize all items in a receipt"""
        # First, try to categorize based on store name
        store_category = self._categorize_by_store(receipt_data.store_name)
        
        # Then categorize each item
        for item in receipt_data.items:
            if not item.category or item.category == "uncategorized":
                item.category = self._categorize_item(item.description, store_category)
        
        return receipt_data
    
    def _categorize_by_store(self, store_name: str) -> Optional[str]:
        """Categorize based on store name"""
        if not store_name:
            return None
        
        store_lower = store_name.lower()
        
        # Check store mappings first
        for store, category in self.rules.get("store_mappings", {}).items():
            if store in store_lower:
                return category
        
        # Check against category keywords
        for category, data in self.rules.get("categories", {}).items():
            keywords = data.get("keywords", [])
            for keyword in keywords:
                if keyword in store_lower:
                    return category
        
        return None
    
    def _categorize_item(self, description: str, store_category: Optional[str] = None) -> str:
        """Categorize a single item based on its description"""
        if not description:
            return "uncategorized"
        
        description_lower = description.lower()
        
        # If we have a store category, use it as a strong hint
        if store_category and store_category != "uncategorized":
            # Check if the item description supports the store category
            category_data = self.rules.get("categories", {}).get(store_category, {})
            keywords = category_data.get("keywords", [])
            
            for keyword in keywords:
                if keyword in description_lower:
                    return store_category
        
        # Score each category based on keyword matches
        category_scores = {}
        
        for category, data in self.rules.get("categories", {}).items():
            score = 0
            keywords = data.get("keywords", [])
            
            for keyword in keywords:
                # Use word boundary matching for more precise categorization
                if re.search(r'\b' + re.escape(keyword) + r'\b', description_lower):
                    # Longer keywords get higher scores
                    score += len(keyword)
                elif keyword in description_lower:
                    # Partial match gets lower score
                    score += len(keyword) // 2
            
            if score > 0:
                category_scores[category] = score
        
        # Return the category with the highest score
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return "uncategorized"
    
    def update_rules(self, new_rules: Dict) -> bool:
        """Update categorization rules"""
        try:
            self.rules = new_rules
            self._save_rules(new_rules)
            return True
        except Exception as e:
            logger.error(f"Failed to update rules: {e}")
            return False
    
    def get_categories(self) -> List[Dict]:
        """Get list of available categories"""
        categories = []
        for name, data in self.rules.get("categories", {}).items():
            categories.append({
                "name": name,
                "description": data.get("description", ""),
                "keyword_count": len(data.get("keywords", []))
            })
        return categories
    
    def add_category(self, name: str, keywords: List[str], description: str = "") -> bool:
        """Add a new category"""
        try:
            if "categories" not in self.rules:
                self.rules["categories"] = {}
            
            self.rules["categories"][name] = {
                "keywords": keywords,
                "description": description
            }
            
            self._save_rules(self.rules)
            return True
        except Exception as e:
            logger.error(f"Failed to add category: {e}")
            return False
    
    def remove_category(self, name: str) -> bool:
        """Remove a category"""
        try:
            if "categories" in self.rules and name in self.rules["categories"]:
                del self.rules["categories"][name]
                self._save_rules(self.rules)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove category: {e}")
            return False
