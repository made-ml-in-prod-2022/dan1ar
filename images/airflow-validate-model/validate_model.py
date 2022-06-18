import os
import click
import json
import pickle
import pandas as pd
import logging
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

logger = logging.getLogger('Validate model')

def calculate_metrics(y_true, y_pred, model_dir):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }
    logger.info("Calculated metrics on validation set.")

    file_path = os.path.join(model_dir, "metrics.json")
    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)
    logger.info(f"Wrote metrics to {file_path!r}.")

def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, "model.pkl")
    with open(file_path, "wb") as file:
        pickle.dump(model, file)
    logger.info(f"Model saved to {file_path!r}.")


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
def val_model(input_dir, model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    logger.info(f"Read model from {model_path!r}.")

    val_features = pd.read_csv(os.path.join(input_dir, "val_features.csv"))
    val_target = pd.read_csv(os.path.join(input_dir, "val_target.csv"))

    pred_target = model.predict(val_features)
    calculate_metrics(val_target, pred_target, model_dir)

if __name__ == "__main__":
    val_model()