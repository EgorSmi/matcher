import json
import pandas as pd
from typing import List, Tuple

from .feature_processor import FeatureProcessor


class SimpleCategoriesFeatures(FeatureProcessor):
    def __init__(self, feature_names: List[str]):
        super().__init__(feature_names)
        self.categories_3_dict = {}
        self.categories_4_dict = {}

    @property
    def processor_name(self) -> str:
        return "SimpleCategoriesFeatures"

    @staticmethod
    def get_cats_from_json(categories_json_string: str) -> Tuple[str, str]:
        categories_dict = json.loads(categories_json_string)
        return categories_dict["3"], categories_dict["4"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        categories = pd.DataFrame()
        categories[["cat_3_name", "cat_4_name"]] = df[["categories"]].apply(
            lambda x: self.get_cats_from_json(x[0]), axis=1, result_type='expand')
        categories_3_unique, categories_4_unique = categories["cat_3_name"].unique(), categories["cat_4_name"].unique()
        self.categories_3_dict, self.categories_4_dict = \
            dict(zip(categories_3_unique, range(len(categories_3_unique)))), \
            dict(zip(categories_4_unique, range(len(categories_4_unique))))
        return df

    def get_categories_encoding(self, df: pd.DataFrame, element_num: str) -> pd.DataFrame:
        categories = pd.DataFrame()
        categories[["cat_3_name", "cat_4_name"]] = df[["categories" + element_num]].apply(
            lambda x: self.get_cats_from_json(x[0]), axis=1, result_type='expand')
        df[["category_" + element_num + "_3", "category_" + element_num + "_4"]] = \
            categories.apply(lambda x: (self.categories_3_dict[x[0]],
                                        self.categories_4_dict[x[1]]),
                             axis=1,
                             result_type='expand')
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.get_categories_encoding(df, "1")
        df = self.get_categories_encoding(df, "2")
        df["categories_match_level"] = df[["category_1_3", "category_1_4", "category_2_3", "category_2_4"]].apply(
            lambda x: int(x[0] == x[2]) + int(x[1] == x[3]), axis=1)
        return df
