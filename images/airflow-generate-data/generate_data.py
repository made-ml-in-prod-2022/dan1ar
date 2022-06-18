import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import click
import logging
import os

logger = logging.getLogger('Data generator')


def save_dataframe(dataframe, save_path: str, filename: str, logger: logging.Logger):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, filename)
    dataframe.to_csv(file_path, index=False)
    logger.info(f"Generated data:: {file_path!r} {dataframe.shape}.")


@click.command("generate_data")
@click.option("--output_dir")
def generate_data(output_dir):
    if output_dir:
        features, targets = make_classification(n_samples=10_000)
        features = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
        targets = pd.DataFrame(targets, columns=['target'])

        save_dataframe(features, output_dir, "data.csv", logger)
        save_dataframe(targets, output_dir, "target.csv", logger)



if __name__ == "__main__":
    generate_data()