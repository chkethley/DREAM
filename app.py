import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import logging
import os

# Assuming your core script is named evolution_engine.py
from evolution_engine import EvolutionEngine, create_model_adapter

# Initialize components
# app = FastAPI()  # We'll initialize it later with dependencies
# engine = EvolutionEngine()  # Don't initialize here

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

API_KEY = os.environ.get("EVOLUTION_ENGINE_API_KEY")
if not API_KEY:
    logger.error("API key not found in environment variable EVOLUTION_ENGINE_API_KEY")
    raise RuntimeError("API key not configured")

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# --- Dependency Injection for EvolutionEngine ---
def get_engine():
    engine = EvolutionEngine("config.yaml") # You might pass a config file path here
    try:
        yield engine
    finally:
        engine.cleanup()

app = FastAPI(dependencies=[Depends(get_engine)])  # Initialize FastAPI here


# Request models
class PredictionRequest(BaseModel):
    model_name: str = Field(..., min_length=1)
    input_text: str = Field(..., min_length=1)

    @validator("model_name")
    def model_name_valid(cls, v):
        allowed_models = ["gpt2", "bert-base-uncased"]  # Load from config ideally
        if v not in allowed_models:
            raise ValueError(f"Invalid model name. Must be one of: {allowed_models}")
        return v

class HyperparameterRequest(BaseModel):
    param_space: dict
    n_calls: int = 20

class PromptRequest(BaseModel):
    task_type: str
    prompt_text: str
    parameters: Optional[dict] = None

# --- Routes ---

@app.get("/")
def health_check():
    return {"message": "Evolution Engine API is running!"}

@app.post("/predict/")
async def predict(request: PredictionRequest, api_key: str = Depends(get_api_key), engine: EvolutionEngine = Depends(get_engine)):
    try:
        # Use the factory to create the appropriate adapter based on request
        model_adapter = create_model_adapter(request.model_name.split('-')[0]) # very simple way to get model type
        model = model_adapter.load_model(request.model_name)
        response = model_adapter.predict(request.input_text)
        return {"prediction": response['prediction']} # Access the prediction
    except ValueError as e:
        logger.error(f"ValueError in prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        logger.exception(f"Unexpected error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/optimize/")
async def optimize_hyperparams(request: HyperparameterRequest, api_key: str = Depends(get_api_key), engine: EvolutionEngine = Depends(get_engine)):
    try:
        engine.hyperparameter_optimizer.param_space = request.param_space
        engine.hyperparameter_optimizer.n_calls = request.n_calls
        optimal_params = engine.run_optimization()
        return {"optimal_params": optimal_params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_prompt/")
async def add_prompt(request: PromptRequest, api_key: str = Depends(get_api_key), engine: EvolutionEngine = Depends(get_engine)):
    try:
        prompt_id = engine.prompt_library.add_prompt(request.task_type, request.prompt_text, request.parameters)
        return {"message": "Prompt added successfully", "prompt_id": prompt_id}
    except Exception as e:
        logger.exception(f"Error adding prompt: {e}")  # Use logger.exception for stack trace
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: int, api_key: str = Depends(get_api_key), engine: EvolutionEngine = Depends(get_engine)):
    try:
        prompt = engine.prompt_library.get_prompt(prompt_id)
        if prompt:
            return prompt
        else:
            raise HTTPException(status_code=404, detail="Prompt not found")
    except Exception as e:
        logger.exception(f"Error retrieving prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/prompts/{prompt_id}")
async def update_prompt(prompt_id: int, request: PromptRequest, api_key: str = Depends(get_api_key), engine: EvolutionEngine = Depends(get_engine)):
    try:
        engine.prompt_library.update_prompt(prompt_id, request.task_type, request.prompt_text, request.parameters)
        return {"message": "Prompt updated successfully"}
    except Exception as e:
        logger.exception(f"Error updating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: int, api_key: str = Depends(get_api_key), engine: EvolutionEngine = Depends(get_engine)):
    try:
        engine.prompt_library.delete_prompt(prompt_id)
        return {"message": "Prompt deleted successfully"}
    except Exception as e:
        logger.exception(f"Error deleting prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Get port from environment, default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
```