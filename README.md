# 🧾 Receipt Tracker API

A production-quality Flask API that processes receipt images and PDFs using OCR, extracts line items, categorizes expenses, and stores data in Excel format. Perfect for expense tracking and financial analysis.

## ✨ Features

- **OCR Processing**: Extract text from images (PNG, JPG, PDF) using Tesseract
- **Smart Parsing**: Automatically extract items, quantities, prices, tax, and totals
- **Auto-Categorization**: Categorize expenses using keyword-based rules
- **Excel Storage**: Store all data in organized Excel workbooks
- **REST API**: Full REST API with comprehensive endpoints
- **Web Interface**: Beautiful HTML upload form for testing
- **Railway Ready**: One-click deployment to Railway with volume mounting

## 🚀 Quick Start

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd receipt-tracker
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr libtesseract-dev poppler-utils
   
   # macOS
   brew install tesseract poppler
   
   # Windows
   # Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

3. **Run the application**:
   ```bash
   python -m app.main
   ```

4. **Access the web interface**: http://localhost:8080

### Railway Deployment

1. **Connect to Railway**:
   - Push your code to GitHub
   - Connect your GitHub repo to Railway
   - Railway will automatically detect the Python app

2. **Configure environment variables**:
   - `DATA_DIR=/data` (default, don't change)
   - `PORT=8080` (default)
   - `HOST=0.0.0.0` (default)

3. **Mount a volume**:
   - In Railway dashboard, add a volume at `/data`
   - This persists your Excel files and uploaded receipts

4. **Deploy**: Railway will automatically build and deploy!

## 📁 Project Structure

```
receipt-tracker/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── main.py              # Main Flask application
│   ├── ocr.py               # OCR processing with Tesseract
│   ├── parser.py            # Receipt parsing logic
│   ├── categorizer.py       # Expense categorization
│   ├── excel_store.py       # Excel file management
│   ├── models.py            # Data models
│   ├── utils.py             # Utility functions
│   ├── rules.default.json   # Default categorization rules
│   └── templates/
│       └── index.html       # Web upload interface
├── tests/
│   ├── test_parser.py       # Parser unit tests
│   └── test_categorizer.py  # Categorizer unit tests
├── requirements.txt         # Python dependencies
├── Procfile                # Railway process file
├── apt.txt                 # System dependencies for Railway
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔌 API Endpoints

### Upload Receipt
```http
POST /api/upload
Content-Type: multipart/form-data

file: [receipt image/PDF]
```

**Response**:
```json
{
  "success": true,
  "message": "Receipt processed successfully",
  "data": {
    "id": "uuid",
    "filename": "receipt.jpg",
    "store_name": "Walmart",
    "transaction_date": "2024-01-15",
    "subtotal": 25.99,
    "tax_amount": 2.08,
    "total_amount": 28.07,
    "items": [
      {
        "description": "Apples",
        "quantity": 2.0,
        "unit_price": 1.50,
        "total_price": 3.00,
        "category": "groceries"
      }
    ],
    "confidence": 0.85
  }
}
```

### Get Transactions
```http
GET /api/transactions?limit=100
```

### Weekly Summary
```http
GET /api/weekly_summary?start=2024-01-01&end=2024-01-07
```

### Update Transaction
```http
POST /api/corrections
Content-Type: application/json

{
  "id": "transaction-uuid",
  "category": "groceries",
  "item_description": "Updated description"
}
```

### Health Check
```http
GET /api/health
```

## 🏷️ Categorization

The system automatically categorizes expenses using keyword-based rules. Categories include:

- **Groceries**: Food, supermarket, produce
- **Restaurants**: Dining, coffee, fast food
- **Gas**: Fuel, gas stations
- **Utilities**: Electric, water, internet
- **Transportation**: Uber, parking, tolls
- **Healthcare**: Pharmacy, medical
- **Entertainment**: Movies, streaming, gym
- **Shopping**: Clothing, electronics, retail
- **Office**: Supplies, business expenses
- **Home**: Hardware, furniture, maintenance

### Customizing Categories

Edit `app/rules.default.json` to add new categories or modify existing ones:

```json
{
  "categories": {
    "new_category": {
      "keywords": ["keyword1", "keyword2"],
      "description": "Description of category"
    }
  },
  "store_mappings": {
    "store_name": "category_name"
  }
}
```

## 📊 Data Storage

All data is stored in Excel format at `/data/export/receipts.xlsx` with the following structure:

| Column | Description |
|--------|-------------|
| id | Unique transaction ID |
| upload_date | When receipt was processed |
| filename | Original file name |
| store_name | Extracted store name |
| transaction_date | Receipt date |
| item_description | Line item description |
| quantity | Item quantity |
| unit_price | Price per unit |
| total_price | Total item price |
| category | Auto-assigned category |
| subtotal | Receipt subtotal |
| tax_amount | Tax amount |
| total_amount | Total receipt amount |
| confidence | OCR confidence score |

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Or run individual test files:

```bash
python -m pytest tests/test_parser.py
python -m pytest tests/test_categorizer.py
```

## 🔧 Configuration

### Environment Variables

- `DATA_DIR`: Directory for data storage (default: `/data`)
- `PORT`: Server port (default: `8080`)
- `HOST`: Server host (default: `0.0.0.0`)
- `FLASK_ENV`: Flask environment (default: `production`)

### OCR Configuration

The OCR system uses Tesseract with optimized settings for receipt processing:
- Preprocessing: Grayscale conversion, contrast enhancement, noise reduction
- OCR Engine: LSTM with custom character whitelist
- Confidence scoring based on receipt characteristics

## 🚀 Railway Deployment Notes

1. **Volume Mounting**: Mount a volume at `/data` to persist Excel files and uploaded receipts
2. **System Dependencies**: The `apt.txt` file installs Tesseract and Poppler automatically
3. **Environment**: Set `DATA_DIR=/data` in Railway environment variables
4. **Scaling**: The app uses Gunicorn with 2 workers by default

## 📈 Performance

- **OCR Processing**: 2-5 seconds per receipt
- **Parsing**: <1 second per receipt
- **Categorization**: <0.1 seconds per receipt
- **Excel Storage**: <1 second per receipt
- **Concurrent Requests**: Supports multiple simultaneous uploads

## 🛠️ Troubleshooting

### Common Issues

1. **OCR not working**: Ensure Tesseract is installed and in PATH
2. **PDF processing fails**: Install Poppler utilities
3. **Excel file not found**: Check DATA_DIR permissions
4. **Low confidence scores**: Try higher resolution images

### Debug Mode

Set `FLASK_ENV=development` for detailed error messages and logging.

## 📝 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub

---

**Built with ❤️ using Flask, Tesseract, and modern Python practices.**
