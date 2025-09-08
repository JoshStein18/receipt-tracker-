from ml_receipt_processor import MLReceiptProcessor

print('=== ML RECEIPT PROCESSOR DEMO ===')
print()

# Initialize ML processor
processor = MLReceiptProcessor()

print('1. ML CATEGORIZATION:')
print('   The ML learns from examples to categorize items:')
test_items = ['apple', 'gas', 'milk', 'computer', 'bread', 'medicine']
for item in test_items:
    category = processor.ml_categorize_item(item)
    print(f'   {item:12} -> {category}')

print()
print('2. ITEM EXTRACTION:')
print('   The ML extracts items from receipt text:')
receipt_text = '''
WALMART STORE
123 Main Street

Apples 2x 1.50
Milk 1x 3.99
Bread 1x 2.50

Subtotal 7.99
Tax 0.80
Total 8.79
'''

items = processor.extract_items_ml(receipt_text)
print(f'   Found {len(items)} items:')
for item in items:
    print(f'   - {item["description"]:15} ${item["total_price"]:6.2f} ({item["category"]})')

print()
print('3. TOTALS EXTRACTION:')
subtotal, tax, total = processor.extract_totals_ml(receipt_text)
print(f'   Subtotal: ${subtotal:.2f}')
print(f'   Tax:      ${tax:.2f}')
print(f'   Total:    ${total:.2f}')

print()
print('4. STORE DETECTION:')
store = processor.extract_store_name_ml(receipt_text)
print(f'   Store: {store}')

print()
print('=== WHY IT MIGHT NOT BE WORKING ===')
print('1. OCR needs Tesseract installed for real images')
print('2. The app needs to be running on the correct port')
print('3. Real receipt images need to be uploaded')
print('4. The ML works on text, but needs OCR to extract text from images')
