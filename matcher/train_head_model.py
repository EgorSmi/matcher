import pandas as pd
import torch

from matcher.features.config import Config
from matcher.feature_register import FeatureRegister
from matcher.head_model.model import CatboostModel
from matcher.head_model.catboost_experiment import CatboostConfig


def train_head_model(
    train_features_df: pd.DataFrame, eval_features_df: pd.DataFrame, all_train_pairs_df: pd.DataFrame,
    label_column: str, save_path: str):
    feature_config = Config()
    feature_register = FeatureRegister(feature_config)

    train_features_df = train_features_df.merge(all_train_pairs_df, on=feature_register.pair_key_columns, how="inner")
    eval_features_df = eval_features_df.merge(all_train_pairs_df, on=feature_register.pair_key_columns, how="inner")

    task_type = "GPU" if torch.cuda.is_available() else "CPU"
    catboost_config = CatboostConfig(task_type)
    catboost_model = CatboostModel(catboost_config)

    catboost_model.fit(
        train_features_df,
        eval_features_df,
        label_column=label_column,
        feature_columns=feature_config.config_feature_names + feature_config.config_embedding_features,
        embedding_feature_columns=feature_config.config_embedding_features,
        output_path=save_path,
    )

    train_pool = CatboostModel.get_pool(train_features_df, label_column,
                                        feature_config.config_feature_names + feature_config.config_embedding_features,
                                        feature_config.config_embedding_features)
    feature_importance = catboost_model.get_feature_importance(train_pool)
    print(f"Feature Importance: ", feature_importance)
