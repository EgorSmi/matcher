import pandas as pd
import numpy as np
import pickle
from typing import List
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist

from .feature_processor import FeatureProcessor
from matcher.config import PICS_COSINE_BALL_TREE_PATH


class PicsCosineFeatures(FeatureProcessor):
    def __init__(self, feature_names: List[str], similarity_threshold: float):
        super().__init__(feature_names)
        with open(PICS_COSINE_BALL_TREE_PATH, 'rb') as f:
            self.useless_pics_tree = pickle.load(f)
        self.similarity_threshold = similarity_threshold

    @property
    def processor_name(self) -> str:
        return "Cosine pics features"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        norm_sum_wo_junk, min_wo_junk = [], []
        for n in range(len(df)):
            print(f'Proceeding pics embeddings: {n}/{len(df)}', end='\r')
            pics_first = [df.loc[n, 'main_pic_embeddings_resnet_v11'][0]]
            pics_second = [df.loc[n, 'main_pic_embeddings_resnet_v12'][0]]
            if df.loc[n, 'pic_embeddings_resnet_v11'] is not None:
                pics_first += df.loc[n, 'pic_embeddings_resnet_v11'].tolist()
            if df.loc[n, 'pic_embeddings_resnet_v12'] is not None:
                pics_second += df.loc[n, 'pic_embeddings_resnet_v12'].tolist()

            pics_first = np.array(pics_first)
            pics_second = np.array(pics_second)

            dist_first, ind_first = self.useless_pics_tree.query(pics_first, k=1)
            pics_first = [pics_first[i] for i in range(len(pics_first)) if
                          not dist_first[i] < self.similarity_threshold]
            dist_second, ind_second = self.useless_pics_tree.query(pics_second, k=1)
            pics_second = [pics_second[i] for i in range(len(pics_second)) if
                           not dist_second[i] < self.similarity_threshold]
            if len(pics_first) == 0 or len(pics_second) == 0:
                norm_sum_wo_junk.append(1)
                min_wo_junk.append(1)
                continue
            distances_all = cdist(pics_first, pics_second, 'cosine')
            if distances_all.shape[0] * distances_all.shape[1] > 1:
                norm_sum_wo_junk.append((np.sum(distances_all) - np.trace(distances_all)) / 4 / (
                        distances_all.shape[0] * distances_all.shape[1] - min(distances_all.shape[0],
                                                                              distances_all.shape[1])))
            else:
                norm_sum_wo_junk.append(np.sum(distances_all))
            min_wo_junk.append(np.min(distances_all))
        df['pics_norm_sum'] = pd.Series(norm_sum_wo_junk)
        df['pics_min'] = pd.Series(min_wo_junk)
        return df
