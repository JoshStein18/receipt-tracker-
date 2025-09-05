from flask import Flask
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__)
    CORS(app)

    # config
    app.config["DATA_DIR"] = os.getenv("DATA_DIR", "./data")
    app.config["UPLOAD_DIR"] = os.path.join(app.config["DATA_DIR"], "uploads")
    app.config["EXPORT_DIR"] = os.path.join(app.config["DATA_DIR"], "export")
    app.config["EXCEL_PATH"] = os.path.join(app.config["EXPORT_DIR"], "receipts.xlsx")

    # ensure dirs
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["EXPORT_DIR"], exist_ok=True)

    return app
