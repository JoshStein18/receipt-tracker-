from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Simple in-memory storage (no database needed)
receipts = []

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
        
        # Create mock receipt data
        receipt_data = {
            'id': str(uuid.uuid4()),
            'filename': file.filename,
            'store_name': 'Sample Store',
            'transaction_date': datetime.now().strftime('%Y-%m-%d'),
            'total_amount': 25.99,
            'subtotal': 23.99,
            'tax_amount': 2.00,
            'items': [
                {
                    'description': 'Sample Item 1',
                    'quantity': 1.0,
                    'unit_price': 12.99,
                    'total_price': 12.99,
                    'category': 'food'
                },
                {
                    'description': 'Sample Item 2',
                    'quantity': 1.0,
                    'unit_price': 11.00,
                    'total_price': 11.00,
                    'category': 'non-food'
                }
            ],
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': 0.9
        }
        
        # Store in memory
        receipts.append(receipt_data)
        
        return jsonify({
            'success': True,
            'message': 'Receipt processed successfully',
            'data': receipt_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to process receipt: {str(e)}'}), 500

@app.route('/api/transactions')
def get_transactions():
    try:
        return jsonify({'success': True, 'data': receipts[-100:]})
    except Exception as e:
        return jsonify({'error': f'Failed to load transactions: {str(e)}'}), 500

@app.route('/api/category_totals')
def get_category_totals():
    try:
        food_total = sum(item['total_price'] for receipt in receipts for item in receipt['items'] if item['category'] == 'food')
        non_food_total = sum(item['total_price'] for receipt in receipts for item in receipt['items'] if item['category'] == 'non-food')
        uncategorized_total = sum(item['total_price'] for receipt in receipts for item in receipt['items'] if item['category'] == 'uncategorized')
        
        food_count = sum(1 for receipt in receipts for item in receipt['items'] if item['category'] == 'food')
        non_food_count = sum(1 for receipt in receipts for item in receipt['items'] if item['category'] == 'non-food')
        uncategorized_count = sum(1 for receipt in receipts for item in receipt['items'] if item['category'] == 'uncategorized')
        
        return jsonify({
            'success': True,
            'data': {
                'food': {'total': float(food_total), 'count': int(food_count)},
                'non_food': {'total': float(non_food_total), 'count': int(non_food_count)},
                'uncategorized': {'total': float(uncategorized_total), 'count': int(uncategorized_count)},
                'grand_total': float(food_total + non_food_total + uncategorized_total)
            }
        })
    except Exception as e:
        return jsonify({'error': f'Failed to calculate totals: {str(e)}'}), 500

@app.route('/api/download_excel')
def download_excel():
    try:
        # Create simple CSV data
        csv_data = "ID,Filename,Store,Date,Item,Category,Price\n"
        for receipt in receipts:
            for item in receipt['items']:
                csv_data += f"{receipt['id']},{receipt['filename']},{receipt['store_name']},{receipt['transaction_date']},{item['description']},{item['category']},{item['total_price']}\n"
        
        return csv_data, 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=receipts.csv'}
    except Exception as e:
        return jsonify({'error': f'Failed to download data: {str(e)}'}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'receipts_count': len(receipts)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)