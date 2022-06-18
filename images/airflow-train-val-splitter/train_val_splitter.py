import click
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import os

logger = logging.getLogger('Train val splitter')


def split_data(features, target, test_size):
    feat_tr, feat_val, tar_tr, tar_val = train_test_split(
        features, target, test_size=test_size, shuffle=True
    )
    logger.info(
        f"Splitted data ({len(features)}) into train/test with "
        f"test_size={test_size}: {len(feat_tr)}/{len(feat_val)}."
    )
    return feat_tr, feat_val, tar_tr, tar_val


@click.command("train_val_split")
@click.option("--input-dir")
@click.option("--val-size", type=float)
def train_val_split(input_dir, val_size):
    features = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    dataframes = split_data(features, target, val_size)
    filenames = [
        "train_features.csv",
        "val_features.csv",
        "train_target.csv",
        "val_target.csv"
    ]

    os.makedirs(input_dir, exist_ok=True)
    for df, filename in zip(dataframes, filenames):
        logger.info(f"Splitted data saved to:: {input_dir}")
        df.to_csv(os.path.join(input_dir, filename), index=False)


if __name__ == "__main__":
    train_val_split()