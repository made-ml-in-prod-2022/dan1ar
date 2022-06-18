import os
import click
import pickle
import xgboost as xgb
import pandas as pd
import logging


logger = logging.getLogger('Train model')

def fit_model(model, features, target):
    model.fit(features, target)
    logger.info("Fitted model.")

def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, "model.pkl")
    with open(file_path, "wb") as file:
        pickle.dump(model, file)
    logger.info(f"Model saved to {file_path!r}.")


@click.command("train_model")
@click.option("--input-dir")
@click.option("--model-dir")
def train_model(input_dir, model_dir):
    train_features = pd.read_csv(os.path.join(input_dir, "train_features.csv"))
    train_target = pd.read_csv(os.path.join(input_dir, "train_target.csv"))
    
    model = xgb.XGBClassifier()

    fit_model(model, train_features, train_target)
    save_model(model, model_dir)

if __name__ == "__main__":
    train_model()