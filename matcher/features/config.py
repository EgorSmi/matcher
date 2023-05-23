from dataclasses import dataclass
from typing import Tuple, List
from .feature_processor import FeatureProcessor
from .matching_bert_feature import MatchingBertFeature
from .simple_categories_features import SimpleCategoriesFeatures
from matcher.config import TEXT_DEBERTA_MATCHER


@dataclass
class Config:
    text_matcher_filepath: str = TEXT_DEBERTA_MATCHER
    feature_processors: Tuple[FeatureProcessor] = (
        SimpleCategoriesFeatures(
            ["category_1_3", "category_1_4",
             "category_2_3", "category_2_4",
             "categories_match_level"]
        ),
        MatchingBertFeature(
            ["matching_bert_score"], TEXT_DEBERTA_MATCHER
        ),
    )

    @property
    def config_feature_names(self) -> List[str]:
        feature_names = []
        for processor in self.feature_processors:
            feature_names.extend(processor.feature_names)
        return feature_names
