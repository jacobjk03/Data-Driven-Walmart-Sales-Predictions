"""FastAPI app to serve Exponential Smoothing forecasts."""
import sys
from pathlib import Path
from typing import Optional

# Allow running as script (e.g. python src/app.py or Code Runner)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR
from src.model import load_model, predict

app = FastAPI(
    title="Walmart Sales Forecasting API",
    description="Store-level sales forecasts using Exponential Smoothing",
    version="1.0.0",
)

# Cache loaded models by store_id
_models: dict[str, object] = {}


def _get_model(store_id: str):
    if store_id in _models:
        return _models[store_id]
    path = MODEL_DIR / f"exp_smoothing_{store_id}.pkl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No model for store_id={store_id}. Train with: python train.py --store-id {store_id}",
        )
    _models[store_id] = load_model(path)
    return _models[store_id]


class PredictRequest(BaseModel):
    store_id: str = Field(..., description="Store identifier (e.g. CA_1, TX_1)")
    steps: int = Field(1, ge=1, le=90, description="Number of days to forecast")


class PredictResponse(BaseModel):
    store_id: str
    steps: int
    forecast: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stores")
def list_stores():
    """List store IDs that have a trained model available."""
    if not MODEL_DIR.exists():
        return {"stores": []}
    stores = []
    for p in MODEL_DIR.glob("exp_smoothing_*.pkl"):
        store_id = p.stem.replace("exp_smoothing_", "")
        stores.append(store_id)
    return {"stores": sorted(stores)}


@app.post("/predict", response_model=PredictResponse)
def forecast(request: PredictRequest):
    """Return sales forecast for the given store and horizon."""
    model = _get_model(request.store_id)
    pred = predict(model, steps=request.steps)
    return PredictResponse(
        store_id=request.store_id,
        steps=request.steps,
        forecast=pred.tolist(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
