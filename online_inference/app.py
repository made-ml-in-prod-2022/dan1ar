import logging
import os
import pickle

from typing import List
import uvicorn
from fastapi import FastAPI
from sklearn.pipeline import Pipeline

from app_pred import make_prediction, TargetModel, TargetResponse

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)

app = FastAPI()

@app.get("/")
def main():
    return "Andreev Danil Homework2"


@app.on_event("startup")
def load_model():
    global model
    print('STARTUP')
    model_path = os.getenv("PATH_TO_MODEL", default='../models/model.pkl')
    print("PATH_TO_MODEL: ", model_path)
    if model_path is None:
        err = f"PATH_TO_MODEL is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/health")
def health() -> int:
    return 200 if not (model is None) else 400


@app.get("/predict", response_model=List[TargetResponse])
def predict(request: TargetModel):
    return make_prediction(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))