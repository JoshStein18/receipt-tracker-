import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

from .models import ReceiptData, TransactionRow, WeeklySummary

logger = logging.getLogger(__name__)

class ExcelStore:
    """Manages Excel file storage for receipt data"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self._ensure_excel_file()
    
    def _ensure_excel_file(self) -> None:
        """Create Excel file with proper structure if it doesn't exist"""
        if not os.path.exists(self.excel_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.excel_path), exist_ok=True)
            
            # Create empty DataFrame with proper columns
            columns = [
                'id', 'upload_date', 'filename', 'store_name', 'transaction_date',
                'item_description', 'quantity', 'unit_price', 'total_price', 'category',
                'subtotal', 'tax_amount', 'total_amount', 'confidence'
            ]
            
            df = pd.DataFrame(columns=columns)
            df.to_excel(self.excel_path, index=False, engine='openpyxl')
            logger.info(f"Created new Excel file: {self.excel_path}")
    
    def append_receipt(self, receipt_data: ReceiptData) -> bool:
        """Append receipt data to Excel file"""
        try:
            # Convert receipt data to transaction rows
            transaction_rows = self._receipt_to_transaction_rows(receipt_data)
            
            # Load existing data
            if os.path.exists(self.excel_path):
                existing_df = pd.read_excel(self.excel_path, engine='openpyxl')
            else:
                existing_df = pd.DataFrame()
            
            # Create new rows DataFrame
            new_rows = []
            for row in transaction_rows:
                            new_rows.append({
                'id': row.id,
                'upload_date': row.upload_date,
                'filename': row.filename,
                'store_name': row.store_name,
                'transaction_date': row.transaction_date,
                'item_description': row.item_description,
                'quantity': row.quantity,
                'unit_price': row.unit_price,
                'total_price': row.total_price,
                'category': row.category,
                'subtotal': row.subtotal,
                'tax_amount': row.tax_amount,
                'total_amount': row.total_amount,
                'confidence': row.confidence,
                'raw_text': receipt_data.raw_text
            })
            
            new_df = pd.DataFrame(new_rows)
            
            # Combine with existing data
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Save to Excel
            combined_df.to_excel(self.excel_path, index=False, engine='openpyxl')
            
            logger.info(f"Successfully appended {len(transaction_rows)} rows to Excel file")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append receipt to Excel: {e}")
            return False
    
    def _receipt_to_transaction_rows(self, receipt_data: ReceiptData) -> List[TransactionRow]:
        """Convert receipt data to transaction rows"""
        rows = []
        
        if not receipt_data.items:
            # Create a single row for the receipt total if no items
            rows.append(TransactionRow(
                id=receipt_data.id,
                upload_date=receipt_data.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
                filename=receipt_data.filename,
                store_name=receipt_data.store_name,
                transaction_date=receipt_data.transaction_date.strftime('%Y-%m-%d') if receipt_data.transaction_date else '',
                item_description='Receipt Total',
                quantity=1.0,
                unit_price=receipt_data.total_amount,
                total_price=receipt_data.total_amount,
                category='uncategorized',
                subtotal=receipt_data.subtotal,
                tax_amount=receipt_data.tax_amount,
                total_amount=receipt_data.total_amount,
                confidence=receipt_data.confidence
            ))
        else:
            # Create a row for each item
            for item in receipt_data.items:
                rows.append(TransactionRow(
                    id=receipt_data.id,
                    upload_date=receipt_data.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
                    filename=receipt_data.filename,
                    store_name=receipt_data.store_name,
                    transaction_date=receipt_data.transaction_date.strftime('%Y-%m-%d') if receipt_data.transaction_date else '',
                    item_description=item.description,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    total_price=item.total_price,
                    category=item.category,
                    subtotal=receipt_data.subtotal,
                    tax_amount=receipt_data.tax_amount,
                    total_amount=receipt_data.total_amount,
                    confidence=item.confidence
                ))
        
        return rows
    
    def get_transactions(self, limit: int = 100) -> List[Dict]:
        """Get latest transactions from Excel file"""
        try:
            if not os.path.exists(self.excel_path):
                return []
            
            df = pd.read_excel(self.excel_path, engine='openpyxl')
            
            if df.empty:
                return []
            
            # Sort by upload_date descending and limit
            df = df.sort_values('upload_date', ascending=False)
            df = df.head(limit)
            
            # Convert to list of dictionaries
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")
            return []
    
    def get_weekly_summary(self, start_date: str, end_date: str) -> WeeklySummary:
        """Get weekly spending summary"""
        try:
            if not os.path.exists(self.excel_path):
                return WeeklySummary(
                    week_start=datetime.strptime(start_date, '%Y-%m-%d'),
                    week_end=datetime.strptime(end_date, '%Y-%m-%d'),
                    total_spent=0.0,
                    category_totals={},
                    transaction_count=0
                )
            
            df = pd.read_excel(self.excel_path, engine='openpyxl')
            
            if df.empty:
                return WeeklySummary(
                    week_start=datetime.strptime(start_date, '%Y-%m-%d'),
                    week_end=datetime.strptime(end_date, '%Y-%m-%d'),
                    total_spent=0.0,
                    category_totals={},
                    transaction_count=0
                )
            
            # Convert date columns to datetime
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
            df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
            
            # Filter by date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Use transaction_date if available, otherwise upload_date
            df['date_to_use'] = df['transaction_date'].fillna(df['upload_date'])
            filtered_df = df[(df['date_to_use'] >= start_dt) & (df['date_to_use'] <= end_dt)]
            
            if filtered_df.empty:
                return WeeklySummary(
                    week_start=start_dt,
                    week_end=end_dt,
                    total_spent=0.0,
                    category_totals={},
                    transaction_count=0
                )
            
            # Calculate totals
            total_spent = filtered_df['total_price'].sum()
            transaction_count = len(filtered_df['id'].unique())
            
            # Calculate category totals
            category_totals = filtered_df.groupby('category')['total_price'].sum().to_dict()
            
            return WeeklySummary(
                week_start=start_dt,
                week_end=end_dt,
                total_spent=total_spent,
                category_totals=category_totals,
                transaction_count=transaction_count
            )
            
        except Exception as e:
            logger.error(f"Failed to get weekly summary: {e}")
            return WeeklySummary(
                week_start=datetime.strptime(start_date, '%Y-%m-%d'),
                week_end=datetime.strptime(end_date, '%Y-%m-%d'),
                total_spent=0.0,
                category_totals={},
                transaction_count=0
            )
    
    def update_transaction(self, transaction_id: str, updates: Dict) -> bool:
        """Update a transaction in the Excel file"""
        try:
            if not os.path.exists(self.excel_path):
                return False
            
            df = pd.read_excel(self.excel_path, engine='openpyxl')
            
            if df.empty:
                return False
            
            # Find the transaction to update
            mask = df['id'] == transaction_id
            if not mask.any():
                return False
            
            # Update the fields
            for field, value in updates.items():
                if field in df.columns:
                    df.loc[mask, field] = value
            
            # Save back to Excel
            df.to_excel(self.excel_path, index=False, engine='openpyxl')
            
            logger.info(f"Successfully updated transaction {transaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update transaction: {e}")
            return False
    
    def get_categories(self) -> List[str]:
        """Get list of all categories used in the data"""
        try:
            if not os.path.exists(self.excel_path):
                return []
            
            df = pd.read_excel(self.excel_path, engine='openpyxl')
            
            if df.empty:
                return []
            
            return df['category'].unique().tolist()
            
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get basic statistics about the data"""
        try:
            if not os.path.exists(self.excel_path):
                return {
                    'total_transactions': 0,
                    'total_spent': 0.0,
                    'categories': [],
                    'date_range': None
                }
            
            df = pd.read_excel(self.excel_path, engine='openpyxl')
            
            if df.empty:
                return {
                    'total_transactions': 0,
                    'total_spent': 0.0,
                    'categories': [],
                    'date_range': None
                }
            
            # Convert date columns
            df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce')
            
            return {
                'total_transactions': len(df['id'].unique()),
                'total_spent': df['total_price'].sum(),
                'categories': df['category'].unique().tolist(),
                'date_range': {
                    'start': df['upload_date'].min().strftime('%Y-%m-%d') if not df['upload_date'].isna().all() else None,
                    'end': df['upload_date'].max().strftime('%Y-%m-%d') if not df['upload_date'].isna().all() else None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                'total_transactions': 0,
                'total_spent': 0.0,
                'categories': [],
                'date_range': None
            }
