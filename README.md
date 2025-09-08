# ML-Powered Receipt Tracker

A machine learning-powered Flask app that intelligently processes receipts, extracts items, and categorizes them using ML algorithms.

## Features

- 🧠 **ML-Powered Item Recognition**: Uses pattern recognition and ML to extract items from receipts
- 🤖 **Intelligent Categorization**: Trained ML model categorizes items as food vs non-food
- 📊 **Advanced OCR**: Multiple Tesseract configurations for better text extraction
- 📈 **Excel Export**: All data saved to Excel with detailed breakdowns
- 🎯 **Smart Parsing**: Extracts store names, dates, totals, and individual items

## ML Capabilities

### Item Extraction
- Pattern recognition for different receipt formats
- Handles "Apples 2x 1.50" and "Milk 3.99" formats
- Skips totals, taxes, and non-item lines

### ML Categorization
- Trained on 92+ examples of food vs non-food items
- Uses TF-IDF vectorization and cosine similarity
- Falls back to keyword matching for confidence
- Examples: apple→food, gas→non-food, computer→non-food

### Text Processing
- OCR with image preprocessing (grayscale, contrast enhancement)
- PDF support with page-by-page processing
- Multiple Tesseract configurations for best results

## API Endpoints

- `POST /api/upload` - Upload receipt image/PDF
- `GET /api/transactions` - Get recent transactions
- `GET /api/category_totals` - Get totals by category
- `GET /api/download_excel` - Download Excel file
- `GET /api/health` - Health check

## Deployment

This app is designed for Railway deployment with:
- `apt.txt` for system dependencies (Tesseract, Poppler)
- `Procfile` for Gunicorn
- `requirements.txt` for Python dependencies

## How It Works

1. **Upload**: User uploads receipt image/PDF
2. **OCR**: Tesseract extracts text with multiple configurations
3. **ML Processing**: Pattern recognition extracts items and prices
4. **Categorization**: ML model categorizes each item
5. **Storage**: Data saved to Excel with full details
6. **Display**: Results shown in dashboard

The ML system is much more intelligent than keyword-only approaches and accurately recognizes items from real receipts!
