import ast
import pandas as pd
import numpy as np
from typing import List
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

from .feature_processor import FeatureProcessor
from matcher.config import ATTRS_BERT_COSINE_MODEL, ATTRS_BERT_COSINE_BATCH_SIZE


class AttrsBertCosineFeatures(FeatureProcessor):
    def __init__(self, feature_names: List[str], values_threshold: float, keys_threshold: float):
        super().__init__(feature_names)
        self.values_threshold = values_threshold
        self.keys_threshold = keys_threshold
        self.model = SentenceTransformer(ATTRS_BERT_COSINE_MODEL)

    @property
    def processor_name(self) -> str:
        return "Bert cosine attrs features"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        result_metrics, cnt_attr1, cnt_attr2, same_keys = [], [], [], []
        for n in range(len(df)):
            print(f'Proceeding attr embeddings: {n}/{len(df)}', end='\r')
            attrs1 = df.loc[n, 'characteristic_attributes_mapping1']
            if attrs1 is None:
                attrs1 = {}
            else:
                attrs1 = ast.literal_eval(attrs1)

            attrs2 = df.loc[n, 'characteristic_attributes_mapping2']
            if attrs2 is None:
                attrs2 = {}
            else:
                attrs2 = ast.literal_eval(attrs2)

            keys1, values1 = [], []
            for i in attrs1.keys():
                if attrs1[i] is not None and i is not None:
                    keys1.append(str(i))
                    values1.append(str(attrs1[i]))

            keys2, values2 = [], []
            for i in attrs2.keys():
                if attrs2[i] is not None and i is not None:
                    keys2.append(str(i))
                    values2.append(str(attrs2[i]))

            embeddings = self.model.encode(keys1 + values1 + keys2 + values2, show_progress_bar=False,
                                           batch_size=ATTRS_BERT_COSINE_BATCH_SIZE)
            keys1_embeddings = embeddings[:len(keys1)]
            values1_embeddings = embeddings[len(keys1):2 * len(keys1)]
            keys2_embeddings = embeddings[2 * len(keys1):2 * len(keys1) + len(keys2)]
            values2_embeddings = embeddings[2 * len(keys1) + len(keys2):]
            distances_keys = cdist(keys1_embeddings, keys2_embeddings, 'cosine') < self.keys_threshold
            cnt_attr1.append(len(keys1))
            cnt_attr2.append(len(keys2))
            cnt_same_keys = np.sum(distances_keys)
            cnt_same_values = 0
            if cnt_same_keys == 0:
                result_metrics.append(0)
                same_keys.append(0)
                continue
            same_keys.append(cnt_same_keys / min(len(keys1), len(keys2)))
            for i in range(len(keys1)):
                for j in range(len(keys2)):
                    if distances_keys[i][j] and \
                            cdist([values1_embeddings[i]], [values2_embeddings[j]], 'cosine')[0][
                                0] < self.values_threshold:
                        cnt_same_values += 1
            result_metrics.append(cnt_same_values / cnt_same_keys)
        df['attr_metric'] = pd.Series(result_metrics)
        df['attr_cnt1'] = pd.Series(cnt_attr1)
        df['attr_cnt2'] = pd.Series(cnt_attr2)
        df['attr_same_keys_proportion'] = pd.Series(same_keys)
        return df
