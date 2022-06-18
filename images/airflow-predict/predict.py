import os
import pandas as pd
import logging
import pickle
import click

logger = logging.getLogger('Predict')

@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def predict(input_dir, output_dir, model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    logger.info(f"Read model from {model_path!r}.")
    
    data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    pred_target = model.predict(data)
    pred_df = pd.DataFrame(pred_target, columns=['pred'])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()