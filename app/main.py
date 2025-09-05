from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from . import create_app
from .ocr import OCRProcessor
from .parser import ReceiptParser
from .categorizer import ReceiptCategorizer
from .excel_store import ExcelStore
from .utils import get_upload_path, safe_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = create_app()

# Initialize components
ocr_processor = OCRProcessor()
receipt_parser = ReceiptParser()
categorizer = ReceiptCategorizer()
excel_store = ExcelStore(app.config["EXCEL_PATH"])

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the upload form"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_receipt():
    """Upload and process a receipt file"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        filename = safe_filename(filename)
        
        # Generate upload path
        upload_path = get_upload_path(app.config["UPLOAD_DIR"], filename)
        
        # Save file
        file.save(upload_path)
        logger.info(f"File saved to: {upload_path}")
        
        # Process the file
        try:
            # Extract text using OCR
            raw_text = ocr_processor.extract_text(upload_path)
            confidence = ocr_processor.get_confidence_score(raw_text)
            
            # Parse the receipt
            receipt_data = receipt_parser.parse_receipt(raw_text, filename)
            receipt_data.confidence = confidence
            
            # Categorize items
            receipt_data = categorizer.categorize_receipt(receipt_data)
            
            # Save to Excel
            success = excel_store.append_receipt(receipt_data)
            
            if not success:
                return jsonify({'error': 'Failed to save to Excel'}), 500
            
            # Prepare response
            response_data = {
                'id': receipt_data.id,
                'filename': receipt_data.filename,
                'store_name': receipt_data.store_name,
                'transaction_date': receipt_data.transaction_date.strftime('%Y-%m-%d') if receipt_data.transaction_date else None,
                'subtotal': receipt_data.subtotal,
                'tax_amount': receipt_data.tax_amount,
                'total_amount': receipt_data.total_amount,
                'items': [
                    {
                        'description': item.description,
                        'quantity': item.quantity,
                        'unit_price': item.unit_price,
                        'total_price': item.total_price,
                        'category': item.category
                    }
                    for item in receipt_data.items
                ],
                'confidence': receipt_data.confidence,
                'raw_text': receipt_data.raw_text[:500] + '...' if len(receipt_data.raw_text) > 500 else receipt_data.raw_text
            }
            
            return jsonify({
                'success': True,
                'message': 'Receipt processed successfully',
                'data': response_data
            })
            
        except Exception as e:
            logger.error(f"Error processing receipt: {e}")
            return jsonify({'error': f'Failed to process receipt: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    """Get latest transactions"""
    try:
        limit = request.args.get('limit', 100, type=int)
        
        # Validate limit
        if limit < 1 or limit > 1000:
            limit = 100
        
        transactions = excel_store.get_transactions(limit)
        
        return jsonify({
            'success': True,
            'data': transactions,
            'count': len(transactions)
        })
        
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return jsonify({'error': 'Failed to get transactions'}), 500

@app.route('/api/weekly_summary', methods=['GET'])
def get_weekly_summary():
    """Get weekly spending summary"""
    try:
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        
        # Validate dates
        if not start_date or not end_date:
            return jsonify({'error': 'start and end dates are required'}), 400
        
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        # Validate date range
        if start_dt > end_dt:
            return jsonify({'error': 'Start date must be before end date'}), 400
        
        # Get summary
        summary = excel_store.get_weekly_summary(start_date, end_date)
        
        response_data = {
            'week_start': summary.week_start.strftime('%Y-%m-%d'),
            'week_end': summary.week_end.strftime('%Y-%m-%d'),
            'total_spent': summary.total_spent,
            'category_totals': summary.category_totals,
            'transaction_count': summary.transaction_count
        }
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        logger.error(f"Error getting weekly summary: {e}")
        return jsonify({'error': 'Failed to get weekly summary'}), 500

@app.route('/api/corrections', methods=['POST'])
def update_transaction():
    """Update transaction data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        transaction_id = data.get('id')
        if not transaction_id:
            return jsonify({'error': 'Transaction ID is required'}), 400
        
        # Prepare updates (only allow certain fields)
        allowed_fields = ['category', 'item_description', 'quantity', 'unit_price', 'total_price']
        updates = {}
        
        for field in allowed_fields:
            if field in data:
                updates[field] = data[field]
        
        if not updates:
            return jsonify({'error': 'No valid fields to update'}), 400
        
        # Update the transaction
        success = excel_store.update_transaction(transaction_id, updates)
        
        if not success:
            return jsonify({'error': 'Transaction not found or update failed'}), 404
        
        return jsonify({
            'success': True,
            'message': 'Transaction updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating transaction: {e}")
        return jsonify({'error': 'Failed to update transaction'}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available categories"""
    try:
        categories = categorizer.get_categories()
        return jsonify({
            'success': True,
            'data': categories
        })
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({'error': 'Failed to get categories'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get basic statistics"""
    try:
        stats = excel_store.get_stats()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_dir': app.config["DATA_DIR"],
        'excel_path': app.config["EXCEL_PATH"]
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
