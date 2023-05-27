import pandas as pd
from typing import List
from torch.utils.data.dataloader import DataLoader

from .feature_processor import FeatureProcessor
from matplotlib import colors
from nltk.stem.snowball import SnowballStemmer
import json


stemmer = SnowballStemmer("russian")
f = open("color_data/rus_to_eng_colors.json")
color_dict = json.load(f)
f = open("color_data/name_to_hex_dict.json")
name_to_hex_dict = json.load(f)


def map_hex_to_rgb(el):
    if el is None:
        return None
    elif isinstance(el, list):
        return [colors.hex2color(hex_c) for hex_c in el]
    return colors.hex2color(el)



class SameColorFeatures(FeatureProcessor):
    def __init__(self, feature_names: List[str]):
        super().__init__(feature_names)

    @property
    def processor_name(self) -> str:
        return "Different color features"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        df["color_parsed"] = df["color_parsed"].apply(lambda x: x[0] if x is not None else None)

        all_colors = set(df.color_parsed.unique())
        all_colors.remove(None)
        all_colors_dict = dict()
        for color in all_colors:
            all_colors_dict[stemmer.stem(color)] = color

        def make_color_from_name(st):
            st = st.lower()
            for color, item in all_colors_dict.items():
                if re.search(color, st):
                    return item
            return None

        df.loc[df.color_parsed.isna(),
               "color_parsed"] = df[df.color_parsed.isna()].name.apply(make_color_from_name)

        df["color_parsed"] = df.color_parsed.apply(lambda x: color_dict[x] if x in color_dict else x)
        df["color_hex"] = df.color_parsed.apply(lambda x: name_to_hex_dict[x] if x in name_to_hex_dict else x)
        df["color_rgb"] = df.color_hex.apply(map_hex_to_rgb)

        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["same_colorname"] = (df['color_parsed1'] == df['color_parsed2']).astype(int)
        df["same_hex"] = (df['color_hex1'] == df['color_hex2']).astype(int)
        df["same_rgb"] = (df['color_rgb1'] == df['color_rgb2']).astype(int)
        return df
