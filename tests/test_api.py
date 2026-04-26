#!/usr/bin/env python3
"""
Test script for Bangkong API functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_api_imports():
    """Test that API modules can be imported."""
    try:
        from bangkong.api.server import app
        print("✓ FastAPI app imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import API modules: {e}")
        return False

def test_deployment_manager():
    """Test that deployment manager works."""
    try:
        from bangkong.deployment.manager import DeploymentManager
        print("✓ Deployment manager imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import deployment manager: {e}")
        return False

if __name__ == "__main__":
    print("Testing Bangkong API implementation...")
    
    success = True
    success &= test_api_imports()
    success &= test_deployment_manager()
    
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        sys.exit(1)