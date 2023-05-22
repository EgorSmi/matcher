from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class FeatureProcessor(ABC):
    def __init__(self):
        pass

    @property
    def processor_name(self) -> str:
        return None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @abstractmethod
    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("Implement me!")
