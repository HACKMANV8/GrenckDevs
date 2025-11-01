#!/usr/bin/env python3
"""
Simple test server to verify Flask is working
"""

from flask import Flask, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='assets', static_url_path='/assets')
CORS(app)

@app.route('/')
def index():
    """Test route"""
    return render_template('genetic_website.html')

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Server is working!',
        'server': 'Flask Test Server'
    })

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'server': 'running',
        'message': 'Test server is operational'
    })

if __name__ == '__main__':
    print("🧪 Starting test server...")
    print("🌐 Test URLs:")
    print("   • Main page: http://localhost:5000")
    print("   • Test API:  http://localhost:5000/test")
    print("   • Health:    http://localhost:5000/api/health")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)