# Bangkong API Server

This directory contains the API server implementation for Bangkong LLM models.

## Features

- FastAPI-based REST API for model inference
- Model deployment utilities
- Health check endpoints
- Configurable deployment targets (local, cloud)

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- PyTorch
- Transformers

## Installation

```bash
pip install -e .[deploy]
```

Or install API dependencies directly:

```bash
pip install -r requirements-api.txt
```

## Usage

### Starting the API Server

```bash
python scripts/start_api.py
```

Or run directly with uvicorn:

```bash
uvicorn bangkong.api.server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- `GET /health` - Health check endpoint
- `GET /model/info` - Get model information
- `POST /generate` - Generate text from prompt

### Deploying Models

```bash
python scripts/deploy.py --model-path /path/to/model --target local
```

### Environment Variables

- `BANGKONG_API_HOST` - API server host (default: 0.0.0.0)
- `BANGKONG_API_PORT` - API server port (default: 8000)
- `BANGKONG_MODEL_PATH` - Path to model file