import pandas as pd
from catboost import Pool
from typing import List

from .model import CatboostModel


class CatBoostScorer:
    def __init__(self, model_path: str = None):
        self.model = CatboostModel.from_pretrained(model_path)

    def get_scores(self, df: pd.DataFrame, feature_names: List[str]) -> List[float]:
        pool = Pool(
            data=df[feature_names],
            feature_names=feature_names,
        )
        preds = self.model.predict(pool, prediction_type="Probability")[:, 1]
        return preds
