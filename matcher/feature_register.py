from matcher.features.config import Config
from typing import List
import pandas as pd
import logging


LOG = logging.getLogger(__name__)


class FeatureRegister:
    def __init__(self, config: Config):
        self.config = config
        self.feature_processors = config.feature_processors

    @property
    def raw_key_columns(self) -> List[str]:
        return ["variantid"]

    @property
    def pair_key_columns(self) -> List[str]:
        return [
            "variantid1",
            "variantid2",
        ]

    @property
    def features(self):
        return self.config.config_feature_names

    def raw_df_prepare(self, df: pd.DataFrame):
        assert len(set(self.raw_key_columns) & set(df.columns)) == len(set(self.raw_key_columns))
        for processor in self.feature_processors:
            LOG.debug(f"{processor.processor_name} preparing single DataFrame..")
            df = processor.preprocess(df)
            LOG.debug(f"{processor.processor_name} has finished single DataFrame preparation..")
        assert len(set(self.raw_key_columns) & set(df.columns)) == len(set(self.raw_key_columns))
        return df

    def compute_pair_features(self, df: pd.DataFrame):
        assert len(set(self.pair_key_columns) & set(df.columns)) == len(set(self.pair_key_columns))
        for processor in self.feature_processors:
            LOG.debug(f"{processor.processor_name} preparing pair feature DataFrame..")
            input_df_columns = set(df.columns)
            df = processor.compute_pair_feature(df)
            LOG.debug(f"{processor.processor_name} has finished pair feature preparation..")
            assert len(input_df_columns) + len(set(processor.feature_names)) == len(set(df.columns))
        assert len(set(self.pair_key_columns) & set(df.columns)) == len(set(self.pair_key_columns))
        return df[[*self.pair_key_columns, *self.features]]
