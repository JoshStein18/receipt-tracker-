import unittest
import json
import tempfile
import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from categorizer import ReceiptCategorizer
from models import ReceiptData, ReceiptItem
from datetime import datetime

class TestReceiptCategorizer(unittest.TestCase):
    """Test cases for the ReceiptCategorizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary rules file for testing
        self.temp_rules_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_rules_file.write(json.dumps({
            "categories": {
                "groceries": {
                    "keywords": ["grocery", "food", "walmart", "supermarket"],
                    "description": "Grocery items"
                },
                "restaurants": {
                    "keywords": ["restaurant", "cafe", "coffee", "starbucks"],
                    "description": "Restaurant expenses"
                },
                "gas": {
                    "keywords": ["gas", "fuel", "shell", "exxon"],
                    "description": "Gas expenses"
                }
            },
            "store_mappings": {
                "walmart": "groceries",
                "starbucks": "restaurants",
                "shell": "gas"
            }
        }))
        self.temp_rules_file.close()
        
        self.categorizer = ReceiptCategorizer(self.temp_rules_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_rules_file.name)
    
    def test_categorize_by_store(self):
        """Test categorization by store name"""
        # Test exact store mapping
        category = self.categorizer._categorize_by_store("Walmart")
        self.assertEqual(category, "groceries")
        
        # Test case insensitive
        category = self.categorizer._categorize_by_store("walmart")
        self.assertEqual(category, "groceries")
        
        # Test partial match
        category = self.categorizer._categorize_by_store("Walmart Superstore")
        self.assertEqual(category, "groceries")
        
        # Test unknown store
        category = self.categorizer._categorize_by_store("Unknown Store")
        self.assertIsNone(category)
    
    def test_categorize_item(self):
        """Test item categorization"""
        # Test grocery item
        category = self.categorizer._categorize_item("Fresh apples")
        self.assertEqual(category, "groceries")
        
        # Test restaurant item
        category = self.categorizer._categorize_item("Coffee drink")
        self.assertEqual(category, "restaurants")
        
        # Test gas item
        category = self.categorizer._categorize_item("Gas fuel")
        self.assertEqual(category, "gas")
        
        # Test unknown item
        category = self.categorizer._categorize_item("Random item")
        self.assertEqual(category, "uncategorized")
    
    def test_categorize_item_with_store_hint(self):
        """Test item categorization with store category hint"""
        # Test with grocery store hint
        category = self.categorizer._categorize_item("Fresh produce", "groceries")
        self.assertEqual(category, "groceries")
        
        # Test with restaurant hint
        category = self.categorizer._categorize_item("Coffee", "restaurants")
        self.assertEqual(category, "restaurants")
    
    def test_categorize_receipt(self):
        """Test full receipt categorization"""
        # Create test receipt data
        receipt_data = ReceiptData(
            id="test-123",
            filename="test.jpg",
            upload_date=datetime.now(),
            store_name="Walmart",
            items=[
                ReceiptItem("Fresh apples", 1.0, 2.99, 2.99),
                ReceiptItem("Milk", 1.0, 3.49, 3.49),
                ReceiptItem("Random item", 1.0, 1.99, 1.99)
            ]
        )
        
        # Categorize the receipt
        categorized_receipt = self.categorizer.categorize_receipt(receipt_data)
        
        # Check that items are categorized
        self.assertEqual(categorized_receipt.items[0].category, "groceries")
        self.assertEqual(categorized_receipt.items[1].category, "groceries")
        self.assertEqual(categorized_receipt.items[2].category, "groceries")  # Store hint applies
    
    def test_get_categories(self):
        """Test getting available categories"""
        categories = self.categorizer.get_categories()
        
        self.assertEqual(len(categories), 3)
        category_names = [cat["name"] for cat in categories]
        self.assertIn("groceries", category_names)
        self.assertIn("restaurants", category_names)
        self.assertIn("gas", category_names)
    
    def test_add_category(self):
        """Test adding a new category"""
        success = self.categorizer.add_category(
            "electronics",
            ["computer", "phone", "laptop"],
            "Electronic devices"
        )
        
        self.assertTrue(success)
        
        # Check that category was added
        categories = self.categorizer.get_categories()
        category_names = [cat["name"] for cat in categories]
        self.assertIn("electronics", category_names)
        
        # Test categorizing with new category
        category = self.categorizer._categorize_item("Computer laptop")
        self.assertEqual(category, "electronics")
    
    def test_remove_category(self):
        """Test removing a category"""
        # First add a category
        self.categorizer.add_category("test", ["test"], "Test category")
        
        # Then remove it
        success = self.categorizer.remove_category("test")
        self.assertTrue(success)
        
        # Check that category was removed
        categories = self.categorizer.get_categories()
        category_names = [cat["name"] for cat in categories]
        self.assertNotIn("test", category_names)
    
    def test_update_rules(self):
        """Test updating categorization rules"""
        new_rules = {
            "categories": {
                "new_category": {
                    "keywords": ["new", "test"],
                    "description": "New category"
                }
            },
            "store_mappings": {
                "new_store": "new_category"
            }
        }
        
        success = self.categorizer.update_rules(new_rules)
        self.assertTrue(success)
        
        # Test that new rules are applied
        category = self.categorizer._categorize_item("New test item")
        self.assertEqual(category, "new_category")
        
        store_category = self.categorizer._categorize_by_store("New Store")
        self.assertEqual(store_category, "new_category")
    
    def test_empty_description_categorization(self):
        """Test categorization with empty description"""
        category = self.categorizer._categorize_item("")
        self.assertEqual(category, "uncategorized")
        
        category = self.categorizer._categorize_item(None)
        self.assertEqual(category, "uncategorized")
    
    def test_keyword_scoring(self):
        """Test that longer keywords get higher scores"""
        # Create a categorizer with overlapping keywords
        test_rules = {
            "categories": {
                "short": {
                    "keywords": ["car"],
                    "description": "Short keyword"
                },
                "long": {
                    "keywords": ["car", "automobile", "vehicle"],
                    "description": "Long keywords"
                }
            },
            "store_mappings": {}
        }
        
        categorizer = ReceiptCategorizer()
        categorizer.rules = test_rules
        
        # "car automobile" should match "long" category better
        category = categorizer._categorize_item("car automobile")
        self.assertEqual(category, "long")
    
    def test_case_insensitive_categorization(self):
        """Test that categorization is case insensitive"""
        category = self.categorizer._categorize_item("GROCERY FOOD")
        self.assertEqual(category, "groceries")
        
        category = self.categorizer._categorize_item("RESTAURANT CAFE")
        self.assertEqual(category, "restaurants")
    
    def test_multiple_keyword_matches(self):
        """Test categorization with multiple keyword matches"""
        category = self.categorizer._categorize_item("grocery food supermarket")
        self.assertEqual(category, "groceries")
        
        # Should return the category with the highest score
        category = self.categorizer._categorize_item("grocery restaurant")
        # This should match groceries due to more keyword matches
        self.assertEqual(category, "groceries")

if __name__ == '__main__':
    unittest.main()
