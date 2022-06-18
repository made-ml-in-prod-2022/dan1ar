from operator import index
import os
import pandas as pd
import click
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('Preprocess')

def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("fill_nan", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return num_pipeline


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir, output_dir):
    if input_dir:
        data = pd.read_csv(os.path.join(input_dir, "data.csv"))
        data = process_numerical_features(data)

        target = pd.read_csv(os.path.join(input_dir, "target.csv"))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Processed data saved to:: {output_dir}")
            data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
            target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    preprocess()