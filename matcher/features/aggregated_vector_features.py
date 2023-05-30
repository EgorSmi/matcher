import pandas as pd
import pickle
from typing import List
from .feature_processor import FeatureProcessor
from matcher.config import RANKED_LIST_PATH


class AggregatedVectorFeatures(FeatureProcessor):
    def __init__(self,feature_names: List[str]):
        super().__init__(feature_names)
        with open(RANKED_LIST_PATH, 'rb') as f:
            self.ranked_list = pickle.load(f)

    @property
    def processor_name(self) -> str:
        return "Aggregated vector features"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        subm=[]
        for i in range(len(df)):
            q=self.ranked_list[df.variantid1[i]]
            try:
                k=(100-q.index(df.variantid2[i]))/100
            except:
                k=0
            subm.append(k)    
        
        df["vector_pred"]=subm
        return df
