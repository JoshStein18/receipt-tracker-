import unittest
from datetime import datetime
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from parser import ReceiptParser
from models import ReceiptData, ReceiptItem

class TestReceiptParser(unittest.TestCase):
    """Test cases for the ReceiptParser class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = ReceiptParser()
    
    def test_parse_simple_receipt(self):
        """Test parsing a simple receipt with basic items"""
        text = """
        WALMART
        123 Main St
        City, State 12345
        
        Item 1                    $5.99
        Item 2                    $3.50
        Item 3                    $2.25
        
        Subtotal                 $11.74
        Tax                      $0.94
        Total                   $12.68
        """
        
        receipt = self.parser.parse_receipt(text, "test_receipt.jpg")
        
        self.assertEqual(receipt.store_name, "WALMART")
        self.assertEqual(len(receipt.items), 3)
        self.assertAlmostEqual(receipt.subtotal, 11.74, places=2)
        self.assertAlmostEqual(receipt.tax_amount, 0.94, places=2)
        self.assertAlmostEqual(receipt.total_amount, 12.68, places=2)
        self.assertGreater(receipt.confidence, 0.5)
    
    def test_parse_receipt_with_quantities(self):
        """Test parsing a receipt with item quantities"""
        text = """
        GROCERY STORE
        
        Apples 2 x 1.50          $3.00
        Milk 1 x 2.99            $2.99
        Bread 3 x 0.99           $2.97
        
        Subtotal                 $8.96
        Tax                      $0.72
        Total                   $9.68
        """
        
        receipt = self.parser.parse_receipt(text, "test_receipt.jpg")
        
        self.assertEqual(len(receipt.items), 3)
        
        # Check first item
        first_item = receipt.items[0]
        self.assertEqual(first_item.description, "Apples")
        self.assertEqual(first_item.quantity, 2.0)
        self.assertEqual(first_item.unit_price, 1.50)
        self.assertEqual(first_item.total_price, 3.00)
    
    def test_parse_receipt_with_date(self):
        """Test parsing a receipt with date information"""
        text = """
        STORE NAME
        01/15/2024
        
        Item 1                   $5.99
        Item 2                   $3.50
        
        Total                   $9.49
        """
        
        receipt = self.parser.parse_receipt(text, "test_receipt.jpg")
        
        self.assertIsNotNone(receipt.transaction_date)
        self.assertEqual(receipt.transaction_date.year, 2024)
        self.assertEqual(receipt.transaction_date.month, 1)
        self.assertEqual(receipt.transaction_date.day, 15)
    
    def test_parse_empty_text(self):
        """Test parsing empty text"""
        receipt = self.parser.parse_receipt("", "test_receipt.jpg")
        
        self.assertEqual(receipt.filename, "test_receipt.jpg")
        self.assertEqual(receipt.store_name, "Unknown Store")
        self.assertEqual(len(receipt.items), 0)
        self.assertEqual(receipt.confidence, 0.0)
    
    def test_parse_malformed_receipt(self):
        """Test parsing malformed receipt text"""
        text = """
        This is not a receipt
        Just some random text
        Without any structure
        """
        
        receipt = self.parser.parse_receipt(text, "test_receipt.jpg")
        
        self.assertEqual(receipt.store_name, "This is not a receipt")
        self.assertEqual(len(receipt.items), 0)
        self.assertLess(receipt.confidence, 0.5)
    
    def test_extract_store_name(self):
        """Test store name extraction"""
        lines = [
            "WALMART SUPERSTORE",
            "123 Main Street",
            "City, State 12345",
            "Phone: 555-1234"
        ]
        
        store_name = self.parser._extract_store_name(lines)
        self.assertEqual(store_name, "WALMART SUPERSTORE")
    
    def test_extract_store_name_skips_dates(self):
        """Test that store name extraction skips date lines"""
        lines = [
            "01/15/2024",
            "12:34:56",
            "WALMART STORE",
            "123 Main Street"
        ]
        
        store_name = self.parser._extract_store_name(lines)
        self.assertEqual(store_name, "WALMART STORE")
    
    def test_extract_totals(self):
        """Test total extraction"""
        lines = [
            "Item 1                    $5.99",
            "Item 2                    $3.50",
            "Subtotal                 $9.49",
            "Tax                      $0.76",
            "Total                   $10.25"
        ]
        
        subtotal, tax, total = self.parser._extract_totals(lines)
        
        self.assertAlmostEqual(subtotal, 9.49, places=2)
        self.assertAlmostEqual(tax, 0.76, places=2)
        self.assertAlmostEqual(total, 10.25, places=2)
    
    def test_extract_items(self):
        """Test item extraction"""
        lines = [
            "WALMART STORE",
            "Item 1                    $5.99",
            "Item 2                    $3.50",
            "Subtotal                 $9.49",
            "Total                   $10.25"
        ]
        
        items = self.parser._extract_items(lines)
        
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].description, "Item 1")
        self.assertEqual(items[0].total_price, 5.99)
        self.assertEqual(items[1].description, "Item 2")
        self.assertEqual(items[1].total_price, 3.50)
    
    def test_parse_item_line_with_quantity(self):
        """Test parsing item line with quantity"""
        line = "Apples 2 x 1.50          $3.00"
        
        item = self.parser._parse_item_line(line)
        
        self.assertIsNotNone(item)
        self.assertEqual(item.description, "Apples")
        self.assertEqual(item.quantity, 2.0)
        self.assertEqual(item.unit_price, 1.50)
        self.assertEqual(item.total_price, 3.00)
    
    def test_parse_item_line_simple(self):
        """Test parsing simple item line"""
        line = "Item 1                    $5.99"
        
        item = self.parser._parse_item_line(line)
        
        self.assertIsNotNone(item)
        self.assertEqual(item.description, "Item 1")
        self.assertEqual(item.quantity, 1.0)
        self.assertEqual(item.unit_price, 5.99)
        self.assertEqual(item.total_price, 5.99)
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        lines = [
            "WALMART STORE",
            "Item 1                    $5.99",
            "Item 2                    $3.50",
            "Subtotal                 $9.49",
            "Tax                      $0.76",
            "Total                   $10.25"
        ]
        
        subtotal = 9.49
        tax = 0.76
        total = 10.25
        items = [ReceiptItem("Item 1", 1.0, 5.99, 5.99)]
        
        confidence = self.parser._calculate_confidence(lines, subtotal, tax, total, items)
        
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)

if __name__ == '__main__':
    unittest.main()
