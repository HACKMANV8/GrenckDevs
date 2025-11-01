#!/usr/bin/env python3
"""
Test script to verify ollama and phi3:latest are working
"""

import ollama

def test_ollama():
    try:
        print("🤖 Testing ollama connection...")
        
        # Create client with explicit host
        client = ollama.Client(host='http://127.0.0.1:11434')
        
        # Test with a simple query
        response = client.chat(model='phi3:latest', messages=[
            {"role": "user", "content": "Hello! Please respond with 'Ollama is working correctly' if you can see this message."}
        ])
        
        print("✅ Ollama Response:")
        print(response['message']['content'])
        print("\n✅ ollama phi3:latest is working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ ollama test failed: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure ollama is running: ollama serve")
        print("2. Check if phi3:latest is installed: ollama list")
        print("3. If not installed: ollama pull phi3:latest")
        return False

if __name__ == "__main__":
    test_ollama()