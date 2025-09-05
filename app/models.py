from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

@dataclass
class ReceiptItem:
    """Represents a single line item from a receipt"""
    description: str
    quantity: float = 1.0
    unit_price: float = 0.0
    total_price: float = 0.0
    category: str = "uncategorized"
    confidence: float = 0.0

@dataclass
class ReceiptData:
    """Represents parsed receipt data"""
    id: str
    filename: str
    upload_date: datetime
    store_name: str
    transaction_date: Optional[datetime] = None
    subtotal: float = 0.0
    tax_amount: float = 0.0
    total_amount: float = 0.0
    items: List[ReceiptItem] = None
    raw_text: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.items is None:
            self.items = []

@dataclass
class WeeklySummary:
    """Represents weekly spending summary"""
    week_start: datetime
    week_end: datetime
    total_spent: float
    category_totals: Dict[str, float]
    transaction_count: int

@dataclass
class TransactionRow:
    """Represents a normalized row for Excel storage"""
    id: str
    upload_date: str
    filename: str
    store_name: str
    transaction_date: str
    item_description: str
    quantity: float
    unit_price: float
    total_price: float
    category: str
    subtotal: float
    tax_amount: float
    total_amount: float
    confidence: float
