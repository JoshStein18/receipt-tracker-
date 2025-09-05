#!/usr/bin/env python3
"""
Startup script for the Receipt Tracker API
"""

import os
import sys
from app.main import app

if __name__ == '__main__':
    # Get configuration from environment
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    print(f"Starting Receipt Tracker API on {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"Data directory: {app.config['DATA_DIR']}")
    
    app.run(host=host, port=port, debug=debug)
