import fire
import logging
import pandas as pd

from matcher.feature_register import FeatureRegister
from matcher.features.config import Config
from matcher.config import HEAD_MODEL_FILENAME, SCORE_COLUMN
from matcher.head_model.catboost_scorer import CatBoostScorer


LOG = logging.getLogger(__name__)


def prepare_features_df(
    test_pairs_filename: str = "test_pairs_wo_target.parquet",
    test_data_filename: str = "test_data.parquet",
) -> pd.DataFrame:
    LOG.info("Reading input dataframes..")
    test_pairs = pd.read_parquet(test_pairs_filename)
    test_etl = pd.read_parquet(test_data_filename)

    config = Config()
    feature_register = FeatureRegister(config)
    LOG.info("Data preprocessing..")
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
    LOG.info("Features calculation..")
    test_features_prepared = feature_register.compute_pair_features(test_features)
    return test_features_prepared


def main(
    test_pairs_filename: str, test_data_filename: str,
    submission_filename: str = "submission.csv",
):
    output_columns = ["variantid1", "variantid2", SCORE_COLUMN]
    test_dataset = prepare_features_df(test_pairs_filename, test_data_filename)
    config = Config()
    head_model = CatBoostScorer(HEAD_MODEL_FILENAME)
    LOG.info("Final model predict..")
    scores = head_model.get_scores(test_dataset, config.config_feature_names, config.config_embedding_features)
    test_dataset[SCORE_COLUMN] = scores
    test_dataset[output_columns].to_csv(submission_filename, index=False)
    LOG.info(f"Submission has been saved to {submission_filename}")


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
        }
    )
