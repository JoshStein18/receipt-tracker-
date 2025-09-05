import re
from typing import List, Optional, Tuple
from datetime import datetime
import logging

from .models import ReceiptData, ReceiptItem
from .utils import extract_currency, parse_date, clean_text

logger = logging.getLogger(__name__)

class ReceiptParser:
    """Parses OCR text to extract receipt data"""
    
    def __init__(self):
        # Common patterns for receipt parsing
        self.currency_pattern = r'\$?(\d+\.?\d{2})'
        self.item_pattern = r'^(.+?)\s+(\d+\.?\d{2})$'
        self.quantity_item_pattern = r'^(.+?)\s+(\d+)\s+x\s+(\d+\.?\d{2})\s+(\d+\.?\d{2})$'
        
        # Keywords that indicate totals, tax, etc.
        self.total_keywords = ['total', 'amount due', 'grand total', 'final total']
        self.subtotal_keywords = ['subtotal', 'sub total', 'sub-total', 'items total']
        self.tax_keywords = ['tax', 'sales tax', 'vat', 'gst', 'hst']
        self.store_keywords = ['store', 'location', 'address', 'phone']
    
    def parse_receipt(self, text: str, filename: str) -> ReceiptData:
        """Parse OCR text into structured receipt data"""
        if not text:
            return self._create_empty_receipt(filename)
        
        # Clean the text
        clean_text_data = clean_text(text)
        lines = [line.strip() for line in clean_text_data.split('\n') if line.strip()]
        
        if not lines:
            return self._create_empty_receipt(filename)
        
        # Extract basic information
        store_name = self._extract_store_name(lines)
        transaction_date = self._extract_date(lines)
        
        # Extract financial data
        subtotal, tax_amount, total_amount = self._extract_totals(lines)
        
        # Extract line items
        items = self._extract_items(lines)
        
        # Calculate confidence based on extracted data
        confidence = self._calculate_confidence(lines, subtotal, tax_amount, total_amount, items)
        
        return ReceiptData(
            id=self._generate_id(),
            filename=filename,
            upload_date=datetime.now(),
            store_name=store_name,
            transaction_date=transaction_date,
            subtotal=subtotal,
            tax_amount=tax_amount,
            total_amount=total_amount,
            items=items,
            raw_text=clean_text_data,
            confidence=confidence
        )
    
    def _create_empty_receipt(self, filename: str) -> ReceiptData:
        """Create empty receipt data when parsing fails"""
        return ReceiptData(
            id=self._generate_id(),
            filename=filename,
            upload_date=datetime.now(),
            store_name="Unknown Store",
            raw_text="",
            confidence=0.0
        )
    
    def _generate_id(self) -> str:
        """Generate unique ID for receipt"""
        import uuid
        return str(uuid.uuid4())
    
    def _extract_store_name(self, lines: List[str]) -> str:
        """Extract store name from receipt lines"""
        # Usually the first few lines contain store information
        for i, line in enumerate(lines[:5]):
            # Skip lines that look like dates, times, or addresses
            if re.match(r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', line):
                continue
            if re.match(r'^\d{1,2}:\d{2}', line):
                continue
            if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', line):  # Phone number
                continue
            
            # If line has reasonable length and doesn't start with numbers, it might be store name
            if 3 <= len(line) <= 50 and not re.match(r'^\d+', line):
                return line
        
        return "Unknown Store"
    
    def _extract_date(self, lines: List[str]) -> Optional[datetime]:
        """Extract transaction date from receipt lines"""
        for line in lines:
            date = parse_date(line)
            if date:
                return date
        return None
    
    def _extract_totals(self, lines: List[str]) -> Tuple[float, float, float]:
        """Extract subtotal, tax, and total amounts"""
        subtotal = 0.0
        tax_amount = 0.0
        total_amount = 0.0
        
        # Look for total amounts (usually at the end)
        for line in reversed(lines[-10:]):  # Check last 10 lines
            line_lower = line.lower()
            
            # Extract amount from line
            amount = extract_currency(line)
            if amount is None:
                continue
            
            # Determine what type of total this is
            if any(keyword in line_lower for keyword in self.total_keywords):
                total_amount = amount
            elif any(keyword in line_lower for keyword in self.subtotal_keywords):
                subtotal = amount
            elif any(keyword in line_lower for keyword in self.tax_keywords):
                tax_amount = amount
        
        # If we found a total but no subtotal, try to calculate it
        if total_amount > 0 and subtotal == 0 and tax_amount > 0:
            subtotal = total_amount - tax_amount
        elif total_amount > 0 and subtotal == 0 and tax_amount == 0:
            # If we only have total, assume it's the subtotal
            subtotal = total_amount
        
        return subtotal, tax_amount, total_amount
    
    def _extract_items(self, lines: List[str]) -> List[ReceiptItem]:
        """Extract line items from receipt"""
        items = []
        
        # Look for lines that match item patterns
        for line in lines:
            # Skip lines that are clearly not items
            if any(keyword in line.lower() for keyword in 
                   self.total_keywords + self.subtotal_keywords + self.tax_keywords + self.store_keywords):
                continue
            
            # Try different item patterns
            item = self._parse_item_line(line)
            if item:
                items.append(item)
        
        return items
    
    def _parse_item_line(self, line: str) -> Optional[ReceiptItem]:
        """Parse a single line to extract item information"""
        # Pattern 1: Description with quantity and price
        match = re.match(self.quantity_item_pattern, line)
        if match:
            description, quantity, unit_price, total_price = match.groups()
            return ReceiptItem(
                description=description.strip(),
                quantity=float(quantity),
                unit_price=float(unit_price),
                total_price=float(total_price),
                confidence=0.8
            )
        
        # Pattern 2: Description with total price only
        match = re.match(self.item_pattern, line)
        if match:
            description, total_price = match.groups()
            return ReceiptItem(
                description=description.strip(),
                quantity=1.0,
                unit_price=float(total_price),
                total_price=float(total_price),
                confidence=0.7
            )
        
        # Pattern 3: Simple description with price at end
        # Look for lines ending with currency amount
        currency_match = re.search(r'\$?(\d+\.?\d{2})$', line)
        if currency_match:
            description = line[:currency_match.start()].strip()
            price = float(currency_match.group(1))
            
            # Skip if description is too short or looks like a total
            if len(description) < 3 or any(keyword in description.lower() for keyword in 
                                         self.total_keywords + self.subtotal_keywords + self.tax_keywords):
                return None
            
            return ReceiptItem(
                description=description,
                quantity=1.0,
                unit_price=price,
                total_price=price,
                confidence=0.6
            )
        
        return None
    
    def _calculate_confidence(self, lines: List[str], subtotal: float, 
                            tax_amount: float, total_amount: float, items: List[ReceiptItem]) -> float:
        """Calculate confidence score for parsed data"""
        confidence_factors = 0
        total_checks = 0
        
        # Check if we found financial data
        if subtotal > 0 or total_amount > 0:
            confidence_factors += 1
        total_checks += 1
        
        # Check if we found items
        if items:
            confidence_factors += 1
        total_checks += 1
        
        # Check if totals make sense
        if subtotal > 0 and tax_amount > 0 and total_amount > 0:
            expected_total = subtotal + tax_amount
            if abs(expected_total - total_amount) < 0.01:  # Within 1 cent
                confidence_factors += 1
        total_checks += 1
        
        # Check if we have reasonable number of lines
        if 5 <= len(lines) <= 100:
            confidence_factors += 1
        total_checks += 1
        
        # Check for currency symbols
        has_currency = any('$' in line for line in lines)
        if has_currency:
            confidence_factors += 1
        total_checks += 1
        
        return confidence_factors / total_checks if total_checks > 0 else 0.0
