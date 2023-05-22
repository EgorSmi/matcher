import fire
import pandas as pd
import numpy as np


from matcher.utils.preprocess import preprocess
from matcher.quality.metric import pr_auc_macro


def prepare_data(
    test_pairs_filename: str = "test_pairs_wo_target.parquet",
    test_data_filename: str = "test_data.parquet",
) -> pd.DataFrame:
    test_pairs = pd.read_parquet(test_pairs_filename)
    test_etl = pd.read_parquet(test_data_filename)

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




def main(test_pairs_filename: str, test_data_filename: str):
    pass



if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
        }
    )