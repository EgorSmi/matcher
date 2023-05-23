import pandas as pd
from pathlib import Path

from matcher.feature_register import FeatureRegister
from matcher.features.config import Config


def prepare_features_df(
    train_features_filename: str = None,
    train_pairs: str = "train_split_pairs.parquet",
    train_data: str = "train_data.parquet",
) -> pd.DataFrame:
    if train_features_filename:
        features_extension = Path(train_features_filename).suffix
        if features_extension == ".parquet":
            features_df = pd.read_parquet(train_features_filename)
        elif features_extension == ".csv":
            features_df = pd.read_csv(train_features_filename)
    train_pairs = pd.read_parquet(train_pairs)
    etl = pd.read_parquet(train_data)

    config = Config()
    feature_register = FeatureRegister(config)

    for feature_processor in feature_register.feature_processors:
        if train_features_filename:
            if len(set(feature_processor.feature_names) & set(features_df.columns)) == len(
                    set(feature_processor.feature_names)):
                continue
        etl = feature_processor.preprocess(etl)

    features = (
        train_pairs
            .merge(
            etl
                .add_suffix('1'),
            on="variantid1"
        )
            .merge(
            etl
                .add_suffix('2'),
            on="variantid2"
        )
    )

    for feature_processor in feature_register.feature_processors:
        if train_features_filename:
            if len(set(feature_processor.feature_names) & set(features_df.columns)) == len(
                    set(feature_processor.feature_names)):
                continue
        features = feature_processor.compute_pair_feature(features)

    if train_features_filename:
        features = features.merge(features_df, on=feature_register.pair_key_columns, how="inner")
    return features
