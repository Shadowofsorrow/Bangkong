"""
FastAPI server for Bangkong LLM model serving
"""

import os
import torch
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..config.schemas import BangkongConfig
from ..models.specialized import create_specialized_model

# Initialize FastAPI app
app = FastAPI(title="Bangkong LLM API", version="0.1.0")

# Global variables for model and tokenizer
model = None
tokenizer = None
config = None


class InferenceRequest(BaseModel):
    """Request model for inference."""
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95


class InferenceResponse(BaseModel):
    """Response model for inference."""
    generated_text: str
    input_length: int
    output_length: int


def load_model(model_path: str):
    """Load model and tokenizer."""
    global model, tokenizer, config
    
    try:
        # Load configuration
        config = BangkongConfig()
        
        # Initialize model
        model = create_specialized_model(config)
        
        # Initialize tokenizer (using default GPT-2 tokenizer for now)
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Load trained weights if they exist
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.environ.get("BANGKONG_MODEL_PATH", "./models/pre_intelligent_model.pt")
    if not load_model(model_path):
        print("Warning: Could not load model on startup")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_loaded": True,
        "model_type": "Bangkong Pre-Intelligent",
        "status": "ready"
    }


@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Generate text from prompt."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(request.prompt, return_tensors="pt")
        input_length = len(inputs[0])
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=True
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_length = len(outputs[0]) - input_length
        
        return InferenceResponse(
            generated_text=generated_text,
            input_length=input_length,
            output_length=output_length
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


def start_server(host: str = "0.0.0.0", port: int = 8000, model_path: str = None):
    """Start the FastAPI server."""
    if model_path:
        load_model(model_path)
    
    uvicorn.run("bangkong.api.server:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    start_server()