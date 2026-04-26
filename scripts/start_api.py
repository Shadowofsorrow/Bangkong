#!/usr/bin/env python3
"""
API server startup script for Bangkong
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.api.server import start_server


def main():
    """Start the API server."""
    # Get host and port from environment variables or use defaults
    host = os.environ.get("BANGKONG_API_HOST", "0.0.0.0")
    port = int(os.environ.get("BANGKONG_API_PORT", 8000))
    model_path = os.environ.get("BANGKONG_MODEL_PATH", None)
    
    print(f"Starting Bangkong API server on {host}:{port}")
    start_server(host=host, port=port, model_path=model_path)


if __name__ == "__main__":
    main()