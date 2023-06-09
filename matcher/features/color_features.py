import pandas as pd
from typing import List
import re
from matplotlib import colors
from nltk.stem.snowball import SnowballStemmer
import json
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from .feature_processor import FeatureProcessor
from matcher.config import RUS_TO_ENG_COLORS, NAME_TO_HEX


def map_hex_to_rgb(el):
    if el is None:
        return None
    elif isinstance(el, list):
        return [colors.hex2color(hex_c) for hex_c in el]
    return colors.hex2color(el)


def color_difference(rgb1: list, rgb2: list):
    def get_sRGBColor(rgb):
        if type(rgb) == list:
            mean_0 = sum([color[0] for color in rgb]) / len(rgb)
            mean_1 = sum([color[1] for color in rgb]) / len(rgb)
            mean_2 = sum([color[2] for color in rgb]) / len(rgb)
            color_rgb = sRGBColor(mean_0, mean_1, mean_2)
        elif type(rgb) == tuple:
            color_rgb = sRGBColor(rgb[0], rgb[1], rgb[2])
        else:
            raise ValueError("Unknown color representation type")
        return color_rgb

    if (rgb1 is not None) and (rgb2 is not None):
        color1 = convert_color(get_sRGBColor(rgb1), LabColor)
        color2 = convert_color(get_sRGBColor(rgb2), LabColor)
        return delta_e_cie2000(color1, color2)
    return -1


class SameColorFeatures(FeatureProcessor):
    def __init__(self, feature_names: List[str]):
        super().__init__(feature_names)
        self.stemmer = SnowballStemmer("russian")
        with open(RUS_TO_ENG_COLORS) as f:
            self.color_dict = json.load(f)
        with open(NAME_TO_HEX) as f:
            self.name_to_hex_dict = json.load(f)

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
            all_colors_dict[self.stemmer.stem(color)] = color

        def make_color_from_name(st):
            st = st.lower()
            for color, item in all_colors_dict.items():
                if re.search(color, st):
                    return item
            return None

        df.loc[df.color_parsed.isna(),"color_parsed"] = df[df.color_parsed.isna()].name.apply(make_color_from_name)

        df["color_parsed"] = df.color_parsed.apply(lambda x: self.color_dict[x] if x in self.color_dict else x)
        df["color_hex"] = df.color_parsed.apply(lambda x: self.name_to_hex_dict[x] if x in self.name_to_hex_dict else x)
        df["color_rgb"] = df.color_hex.apply(map_hex_to_rgb)

        return df


    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["same_colorname"] = (df["color_parsed1"] == df["color_parsed2"]).astype(int)
        df["same_hex"] = (df["color_hex1"] == df["color_hex2"]).astype(int)
        df["same_rgb"] = (df["color_rgb1"] == df["color_rgb2"]).astype(int)
        df["color_difference_ciede"] = df.apply(lambda x: color_difference(x["color_rgb1"], x["color_rgb2"]), axis=1)
        return df
