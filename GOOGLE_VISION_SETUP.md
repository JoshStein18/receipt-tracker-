# Google Vision API Setup Guide

## 🚀 **Real Machine Learning with Google Vision API**

This app now uses **Google Vision API** for real OCR and intelligent text processing.

### **What Google Vision API Provides:**

1. **🧠 Real OCR:** Advanced text detection from images
2. **📊 Intelligent Processing:** Better text extraction than Tesseract
3. **🎯 High Accuracy:** Google's ML models trained on millions of images
4. **📱 Multi-format Support:** Images, PDFs, handwritten text
5. **🌐 Cloud-based:** No local dependencies to install

### **Setup Steps:**

#### **1. Create Google Cloud Project**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Vision API

#### **2. Create Service Account**
1. Go to IAM & Admin → Service Accounts
2. Create new service account
3. Download the JSON key file
4. Set environment variable: `GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json`

#### **3. Enable Billing**
- Google Vision API requires billing to be enabled
- First 1,000 requests per month are free
- After that: $1.50 per 1,000 requests

#### **4. Deploy to Railway**
1. Add the JSON key file to your Railway project
2. Set environment variable in Railway dashboard
3. Deploy the app

### **Environment Variables:**
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id
```

### **Benefits Over Previous "ML":**
- ✅ **Real machine learning** (not keyword matching)
- ✅ **Google's trained models** (millions of images)
- ✅ **Higher accuracy** for receipt processing
- ✅ **Better text extraction** from poor quality images
- ✅ **Handles multiple languages** and formats
- ✅ **No local dependencies** to install

### **Cost:**
- **Free tier:** 1,000 requests/month
- **Paid:** $1.50 per 1,000 requests
- **Typical usage:** $5-20/month for personal use

### **Fallback:**
If Google Vision API is not available, the app falls back to mock data processing, so it will still work but with sample data instead of real OCR.
