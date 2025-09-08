from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
from datetime import datetime
import uuid
import json
import logging

from smart_ml_processor import SmartMLProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXCEL_FILE'] = 'receipts.xlsx'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize smart ML processor for any receipt format
receipt_processor = SmartMLProcessor()

def save_to_excel(data):
    """Save receipt data to Excel file"""
    try:
        # Create new rows for each item
        new_rows = []
        for item in data['items']:
            new_row = {
                'id': str(uuid.uuid4()),
                'upload_date': data['upload_date'],
                'filename': data['filename'],
                'store_name': data['store_name'],
                'transaction_date': data['transaction_date'],
                'item_description': item['description'],
                'quantity': item['quantity'],
                'unit_price': item['unit_price'],
                'total_price': item['total_price'],
                'category': item['category'],
                'subtotal': data['subtotal'],
                'tax_amount': data['tax_amount'],
                'total_amount': data['total_amount'],
                'confidence': data['confidence']
            }
            new_rows.append(new_row)
        
        # If no items, create a single row
        if not new_rows:
            new_rows.append({
                'id': str(uuid.uuid4()),
                'upload_date': data['upload_date'],
                'filename': data['filename'],
                'store_name': data['store_name'],
                'transaction_date': data['transaction_date'],
                'item_description': 'No items extracted',
                'quantity': 1.0,
                'unit_price': data['total_amount'],
                'total_price': data['total_amount'],
                'category': 'uncategorized',
                'subtotal': data['subtotal'],
                'tax_amount': data['tax_amount'],
                'total_amount': data['total_amount'],
                'confidence': data['confidence']
            })
        
        # Load existing data or create new
        if os.path.exists(app.config['EXCEL_FILE']):
            df = pd.read_excel(app.config['EXCEL_FILE'])
        else:
            df = pd.DataFrame(columns=[
                'id', 'upload_date', 'filename', 'store_name', 'transaction_date',
                'item_description', 'quantity', 'unit_price', 'total_price',
                'category', 'subtotal', 'tax_amount', 'total_amount', 'confidence'
            ])
        
        # Add new rows
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        
        # Save to Excel
        df.to_excel(app.config['EXCEL_FILE'], index=False)
        logger.info(f"Saved {len(new_rows)} rows to Excel")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to Excel: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_receipt():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        logger.info(f"File saved: {file_path}")
        
        # Process with smart ML model (handles any receipt format)
        try:
            parsed_data = receipt_processor.process_receipt(file_path, filename)
            logger.info(f"Smart ML processing complete: {len(parsed_data['items'])} items extracted")
            
            # Save to Excel
            save_success = save_to_excel(parsed_data)
            if not save_success:
                logger.warning("Failed to save to Excel")
            
            return jsonify({
                'success': True,
                'message': 'Receipt processed successfully with smart ML model',
                'data': parsed_data
            })
            
        except Exception as e:
            logger.error(f"Receipt processing failed: {e}")
            return jsonify({'error': f'Receipt processing failed: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': f'Failed to process receipt: {str(e)}'}), 500

@app.route('/api/transactions')
def get_transactions():
    try:
        if os.path.exists(app.config['EXCEL_FILE']):
            df = pd.read_excel(app.config['EXCEL_FILE'])
            # Return last 100 transactions
            transactions = df.tail(100).to_dict('records')
            return jsonify({'success': True, 'data': transactions})
        else:
            return jsonify({'success': True, 'data': []})
    except Exception as e:
        logger.error(f"Failed to load transactions: {e}")
        return jsonify({'error': f'Failed to load transactions: {str(e)}'}), 500

@app.route('/api/category_totals')
def get_category_totals():
    try:
        if os.path.exists(app.config['EXCEL_FILE']):
            df = pd.read_excel(app.config['EXCEL_FILE'])
            
            # Calculate totals by category
            food_total = df[df['category'] == 'food']['total_price'].sum()
            non_food_total = df[df['category'] == 'non-food']['total_price'].sum()
            uncategorized_total = df[df['category'] == 'uncategorized']['total_price'].sum()
            
            food_count = len(df[df['category'] == 'food'])
            non_food_count = len(df[df['category'] == 'non-food'])
            uncategorized_count = len(df[df['category'] == 'uncategorized'])
            
            return jsonify({
                'success': True,
                'data': {
                    'food': {'total': float(food_total), 'count': int(food_count)},
                    'non_food': {'total': float(non_food_total), 'count': int(non_food_count)},
                    'uncategorized': {'total': float(uncategorized_total), 'count': int(uncategorized_count)},
                    'grand_total': float(food_total + non_food_total + uncategorized_total)
                }
            })
        else:
            return jsonify({
                'success': True,
                'data': {
                    'food': {'total': 0.0, 'count': 0},
                    'non_food': {'total': 0.0, 'count': 0},
                    'uncategorized': {'total': 0.0, 'count': 0},
                    'grand_total': 0.0
                }
            })
    except Exception as e:
        logger.error(f"Failed to calculate totals: {e}")
        return jsonify({'error': f'Failed to calculate totals: {str(e)}'}), 500

@app.route('/api/download_excel')
def download_excel():
    try:
        if os.path.exists(app.config['EXCEL_FILE']):
            return send_from_directory('.', app.config['EXCEL_FILE'], as_attachment=True)
        else:
            return jsonify({'error': 'Excel file not found'}), 404
    except Exception as e:
        logger.error(f"Failed to download Excel: {e}")
        return jsonify({'error': f'Failed to download Excel: {str(e)}'}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'excel_file': app.config['EXCEL_FILE'],
        'features': 'Smart ML (Any Receipt) + Google Vision + Excel Export'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)