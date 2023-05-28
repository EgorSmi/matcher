from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class Bert64EmbeddingFeatures(ABC):
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    @property
    def processor_name(self) -> str:
        return "Bert64EmbeddingFeatures"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # no need to do smth
