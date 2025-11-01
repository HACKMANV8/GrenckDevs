#!/usr/bin/env python3
"""
ngrok Setup Script for GrenckDevs Genetic Analysis Platform
This script will help you configure ngrok with your auth token
"""

import os
import sys
from pyngrok import ngrok, conf

def setup_ngrok():
    """Setup ngrok with auth token"""
    print("🔧 ngrok Setup for GrenckDevs Genetic Analysis Platform")
    print("="*60)
    
    print("📋 Steps to get your ngrok auth token:")
    print("1. Go to: https://dashboard.ngrok.com/signup")
    print("2. Create a free account")
    print("3. Go to: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("4. Copy your auth token")
    print("="*60)
    
    # Get auth token from user
    auth_token = input("🔑 Enter your ngrok auth token: ").strip()
    
    if not auth_token:
        print("❌ No auth token provided. Exiting...")
        return False
    
    try:
        # Set the auth token
        ngrok.set_auth_token(auth_token)
        print("✅ Auth token configured successfully!")
        
        # Test the connection
        print("🧪 Testing ngrok connection...")
        tunnel = ngrok.connect(8080)  # Test tunnel
        print(f"✅ Test successful! Tunnel created: {tunnel.public_url}")
        ngrok.disconnect(tunnel.public_url)
        print("✅ Test tunnel closed")
        
        print("="*60)
        print("🎉 ngrok is now configured and ready to use!")
        print("🚀 You can now run: python deploy_ngrok.py")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to configure ngrok: {str(e)}")
        print("💡 Please check your auth token and try again")
        return False

if __name__ == "__main__":
    setup_ngrok()