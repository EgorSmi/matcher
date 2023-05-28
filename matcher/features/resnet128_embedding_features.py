from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class Resnet128EmbeddingFeatures(ABC):
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    @property
    def processor_name(self) -> str:
        return "Resnet128EmbeddingFeatures"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["main_pic_embeddings_resnet_v11"] = df["main_pic_embeddings_resnet_v11"].apply(lambda x: x[0])
        df["main_pic_embeddings_resnet_v12"] = df["main_pic_embeddings_resnet_v12"].apply(lambda x: x[0])
        return df
