import fire
import pandas as pd

from matcher.feature_register import FeatureRegister
from matcher.features.config import Config


def prepare_features_df(
    test_pairs_filename: str = "test_pairs_wo_target.parquet",
    test_data_filename: str = "test_data.parquet",
) -> pd.DataFrame:
    test_pairs = pd.read_parquet(test_pairs_filename)
    test_etl = pd.read_parquet(test_data_filename)

    config = Config()
    feature_register = FeatureRegister(config)
    test_etl = feature_register.raw_df_prepare(test_etl)

    test_features = (
        test_pairs
            .merge(
            test_etl
                .add_suffix('1'),
            on="variantid1"
        )
            .merge(
            test_etl
                .add_suffix('2'),
            on="variantid2"
        )
    )

    test_features_prepared = feature_register.compute_pair_features(test_features)
    return test_features_prepared



def main(test_pairs_filename: str, test_data_filename: str):
    pass



if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
        }
    )
