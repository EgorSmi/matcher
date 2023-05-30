import json
import pandas as pd
from typing import List
from .feature_processor import FeatureProcessor


def attrs_to_set(st):
    s = set()
    if st is not None:
        st = json.loads(st)
        for key, item in st.items():
            for it in item:
                s.add(key + '___' + it.strip().lower())
    return s


def find_iou(set_attrs1, set_attrs2):
    try:
        i = len(set_attrs1.intersection(set_attrs2))
        u = len(set_attrs1.union(set_attrs2))
        return i / (u + 1e-5)
    except:
        return 0


class AttrsIoUFeature(FeatureProcessor):
    def __init__(self, feature_names: List[str]):
        super().__init__(feature_names)

    def processor_name(self) -> str:
        return "AttrsIoUFeature"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df["set_attrs"] = df["characteristic_attributes_mapping"].map(attrs_to_set)
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["attrs_iou"] = df.apply(lambda x: find_iou(x["set_attrs1"], x["set_attrs2"]), axis=1)
        return df
