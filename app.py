from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
from datetime import datetime
import uuid
import json
import logging

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

# Simple categorization rules
FOOD_KEYWORDS = [
    'food', 'eat', 'meal', 'snack', 'drink', 'beverage', 'coffee', 'tea',
    'breakfast', 'lunch', 'dinner', 'cafe', 'restaurant', 'dining',
    'pizza', 'burger', 'sandwich', 'salad', 'soup', 'pasta', 'chicken',
    'beef', 'fish', 'vegetarian', 'vegan', 'organic', 'fresh', 'produce',
    'apples', 'bananas', 'oranges', 'grapes', 'strawberries', 'milk',
    'cheese', 'yogurt', 'butter', 'eggs', 'bread', 'cereal', 'rice',
    'grocery', 'supermarket', 'market', 'dairy', 'fruit', 'vegetable',
    'walmart', 'safeway', 'kroger', 'whole foods', 'costco',
    'mcdonald', 'burger king', 'subway', 'starbucks', 'dunkin',
    'chipotle', 'taco bell', 'kfc', 'pizza hut', 'domino'
]

NON_FOOD_KEYWORDS = [
    'gas', 'fuel', 'gasoline', 'station', 'shell', 'exxon', 'chevron',
    'electric', 'water', 'utility', 'power', 'energy', 'internet', 'phone',
    'uber', 'lyft', 'taxi', 'bus', 'train', 'metro', 'transit', 'parking',
    'pharmacy', 'drug', 'medicine', 'medical', 'doctor', 'hospital',
    'cvs', 'walgreens', 'rite aid', 'health', 'prescription',
    'movie', 'cinema', 'theater', 'netflix', 'spotify', 'amazon prime',
    'entertainment', 'sports', 'gym', 'fitness', 'club',
    'clothing', 'apparel', 'shoes', 'accessories', 'electronics', 'amazon',
    'office', 'supplies', 'stationery', 'paper', 'pen', 'pencil', 'staples',
    'home', 'depot', 'lowes', 'hardware', 'furniture', 'appliance',
    'computer', 'laptop', 'desktop', 'tablet', 'phone', 'smartphone', 'tech',
    'software', 'hardware', 'cable', 'charger', 'battery', 'headphones',
    'monitor', 'keyboard', 'mouse', 'webcam', 'camera', 'gaming',
    'apple', 'samsung', 'microsoft', 'google', 'best buy', 'dell', 'hp'
]

def categorize_item(description, store_name=""):
    """Simple categorization based on keywords"""
    text = (description + " " + store_name).lower()
    
    food_score = sum(1 for keyword in FOOD_KEYWORDS if keyword in text)
    non_food_score = sum(1 for keyword in NON_FOOD_KEYWORDS if keyword in text)
    
    if food_score > non_food_score:
        return "food"
    elif non_food_score > food_score:
        return "non-food"
    else:
        return "uncategorized"

def parse_receipt_text(text):
    """Simple text parsing - extract basic info"""
    lines = text.strip().split('\n')
    
    # Find store name (usually first line)
    store_name = lines[0].strip() if lines else "Unknown Store"
    
    # Find date (look for date patterns)
    transaction_date = None
    for line in lines:
        if any(char.isdigit() for char in line) and ('/' in line or '-' in line):
            transaction_date = line.strip()
            break
    
    # Find total amount (look for numbers with $ or at end of lines)
    total_amount = 0.0
    for line in lines:
        if '$' in line:
            # Extract number after $
            import re
            numbers = re.findall(r'\$?(\d+\.?\d*)', line)
            if numbers:
                try:
                    total_amount = float(numbers[-1])  # Take last number
                    break
                except:
                    continue
    
    # Create a simple item from the total
    items = []
    if total_amount > 0:
        items.append({
            'description': 'Receipt Item',
            'quantity': 1.0,
            'unit_price': total_amount,
            'total_price': total_amount,
            'category': categorize_item('Receipt Item', store_name)
        })
    
    return {
        'store_name': store_name,
        'transaction_date': transaction_date,
        'total_amount': total_amount,
        'subtotal': total_amount * 0.9,  # Estimate
        'tax_amount': total_amount * 0.1,  # Estimate
        'items': items
    }

def save_to_excel(data):
    """Save receipt data to Excel file"""
    try:
        # Create new row
        new_row = {
            'id': str(uuid.uuid4()),
            'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': data['filename'],
            'store_name': data['store_name'],
            'transaction_date': data['transaction_date'],
            'item_description': data['items'][0]['description'] if data['items'] else 'No items',
            'quantity': data['items'][0]['quantity'] if data['items'] else 1.0,
            'unit_price': data['items'][0]['unit_price'] if data['items'] else 0.0,
            'total_price': data['items'][0]['total_price'] if data['items'] else 0.0,
            'category': data['items'][0]['category'] if data['items'] else 'uncategorized',
            'subtotal': data['subtotal'],
            'tax_amount': data['tax_amount'],
            'total_amount': data['total_amount'],
            'confidence': 0.8
        }
        
        # Load existing data or create new
        if os.path.exists(app.config['EXCEL_FILE']):
            df = pd.read_excel(app.config['EXCEL_FILE'])
        else:
            df = pd.DataFrame(columns=[
                'id', 'upload_date', 'filename', 'store_name', 'transaction_date',
                'item_description', 'quantity', 'unit_price', 'total_price',
                'category', 'subtotal', 'tax_amount', 'total_amount', 'confidence'
            ])
        
        # Add new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to Excel
        df.to_excel(app.config['EXCEL_FILE'], index=False)
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
        
        # For now, create mock data (we'll add OCR later)
        mock_text = f"""
        {filename.replace('.', ' ').replace('_', ' ').title()}
        
        Receipt Date: {datetime.now().strftime('%Y-%m-%d')}
        
        Item 1                    $10.00
        Item 2                    $15.50
        Item 3                    $8.75
        
        Subtotal                  $34.25
        Tax                       $3.43
        Total                     $37.68
        """
        
        # Parse the mock data
        parsed_data = parse_receipt_text(mock_text)
        parsed_data['filename'] = filename
        parsed_data['raw_text'] = mock_text
        
        # Save to Excel
        save_success = save_to_excel(parsed_data)
        if not save_success:
            logger.warning("Failed to save to Excel")
        
        return jsonify({
            'success': True,
            'message': 'Receipt processed successfully',
            'data': parsed_data
        })
        
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
        'excel_file': app.config['EXCEL_FILE']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)