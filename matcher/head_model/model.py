from catboost import Pool, CatBoost
import pandas as pd
from typing import List

from matcher.head_model.catboost_experiment import CatboostConfig


class CatboostModel:
    def __init__(self, config: CatboostConfig):
        self.config = config
        self.model = None

    @classmethod
    def from_pretrained(cls, model_path: str):
        path_extension = model_path.split(".")[-1]
        if path_extension != "cbm":
            raise ValueError("Model path should be in .cbm format!")
        booster = CatBoost()
        booster.load_model(model_path, format="cbm")
        return booster

    def _save_model(self, output_path: str):
        if self.model is None:
            raise RuntimeError("Run .fit() at first!")

        self.model.save_model(output_path, format="cbm")

    @staticmethod
    def get_pool(df: pd.DataFrame, label_column: str, feature_columns: List[str], group_column: str = None) -> Pool:
        if group_column:
            pool = Pool(
                data=df[feature_columns],
                label=df[label_column],
                feature_names=list(feature_columns),
                group_id=df[group_column],
            )
        else:
            pool = Pool(
                data=df[feature_columns],
                label=df[label_column],
                feature_names=list(feature_columns),
            )
        return pool

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        label_column: str,
        feature_columns: List[str],
        output_path: str,
        group_column: str = None,
    ):
        train_pool = CatboostModel.get_pool(train_df, label_column, feature_columns, group_column)
        eval_pool = CatboostModel.get_pool(eval_df, label_column, feature_columns, group_column)
        catboost_params = self.config.catboost_params(feature_names=train_pool.get_feature_names())
        self.model = CatBoost(catboost_params)
        self.model.fit(train_pool, eval_set=eval_pool, verbose_eval=True)
        self._save_model(output_path)

    def predict_proba(self, pool: Pool):
        if self.model is None:
            raise RuntimeError("Load model at first!")

        return self.model.predict(pool, prediction_type="Probability")[:, 1]

    def get_feature_importance(self, pool: Pool):
        if self.model is None:
            raise RuntimeError("Run .fit() at first!")
        importances = self.model.get_feature_importance(pool).tolist()
        importance_dict = {}
        for name, importance in zip(pool.get_feature_names(), importances):
            importance_dict[name] = importance
        importance_dict = sorted(importance_dict.items(), key=lambda x: -x[1])
        return importance_dict
