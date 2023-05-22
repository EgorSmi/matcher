from dataclasses import dataclass
from typing import Tuple, List
from .feature_processor import FeatureProcessor
from .matching_bert_feature import MatchingBertFeature
from matcher.config import TEXT_DEBERTA_MATCHER


@dataclass
class Config:
    text_matcher_filepath: str = TEXT_DEBERTA_MATCHER
    feature_processors: Tuple[FeatureProcessor] = (
        MatchingBertFeature(
            ["matching_bert_score"], TEXT_DEBERTA_MATCHER
        )
    )

    @property
    def config_feature_names(self) -> List[str]:
        feature_names = []
        for processor in self.feature_processors:
            feature_names.extend(processor.feature_names)
        return feature_names
