from ml_receipt_processor import MLReceiptProcessor

# Test ML processor
processor = MLReceiptProcessor()

# Test categorization
test_items = ['apple', 'gas', 'milk', 'computer', 'bread', 'medicine']
print("Testing ML categorization:")
for item in test_items:
    category = processor.ml_categorize_item(item)
    print(f'{item} -> {category}')

# Test item extraction
test_text = '''
WALMART STORE
123 Main Street

Apples 2x 1.50
Milk 1x 3.99
Bread 1x 2.50

Subtotal 7.99
Tax 0.80
Total 8.79
'''

print(f'\nTesting item extraction:')
items = processor.extract_items_ml(test_text)
print(f'Extracted {len(items)} items:')
for item in items:
    print(f'  {item["description"]} - ${item["total_price"]:.2f} - {item["category"]}')

# Test totals extraction
subtotal, tax, total = processor.extract_totals_ml(test_text)
print(f'\nTotals: Subtotal=${subtotal:.2f}, Tax=${tax:.2f}, Total=${total:.2f}')

# Test store name extraction
store = processor.extract_store_name_ml(test_text)
print(f'Store: {store}')
