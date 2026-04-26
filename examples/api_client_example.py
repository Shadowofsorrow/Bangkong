#!/usr/bin/env python3
"""
Example script showing how to use the Bangkong API client
"""

import requests
import json

def test_api_client():
    """Example client code for the Bangkong API."""
    
    # API endpoint
    base_url = "http://localhost:8000"
    
    # Health check
    try:
        response = requests.get(f"{base_url}/health")
        print("Health check:", response.json())
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Model info
    try:
        response = requests.get(f"{base_url}/model/info")
        print("Model info:", response.json())
    except Exception as e:
        print(f"Model info failed: {e}")
    
    # Text generation example
    try:
        generation_data = {
            "prompt": "The future of artificial intelligence",
            "max_length": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{base_url}/generate",
            json=generation_data
        )
        print("Generation result:", response.json())
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    test_api_client()