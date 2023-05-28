import pandas as pd
from catboost import Pool
from typing import List

from .model import CatboostModel


class CatBoostScorer:
    def __init__(self, model_path: str = None):
        self.model = CatboostModel.from_pretrained(model_path)

    def get_scores(
        self, df: pd.DataFrame, feature_names: List[str], embedding_feature_columns: List[str]
    ) -> List[float]:
        pool = Pool(
            data=df[feature_names],
            feature_names=feature_names,
            embedding_features=embedding_feature_columns,
        )
        preds = self.model.predict(pool, prediction_type="Probability")[:, 1]
        return preds
