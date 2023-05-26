from dataclasses import dataclass
from typing import Tuple, List
from .feature_processor import FeatureProcessor
from .matching_bert_feature import MatchingBertFeature
from .simple_categories_features import SimpleCategoriesFeatures
from .pics_cosine_features import PicsCosineFeatures
from .attrs_bert_cosine_features import AttrsBertCosineFeatures
from matcher.config import TEXT_DEBERTA_MATCHER, PICS_COSINE_SIMILARITY_THRESHOLD, ATTRS_BERT_COSINE_KEYS_THRESHOLD, \
    ATTRS_BERT_COSINE_VALUES_THRESHOLD


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
        PicsCosineFeatures(
            ['pics_norm_sum', 'pics_min'], PICS_COSINE_SIMILARITY_THRESHOLD
        ),
        AttrsBertCosineFeatures(
            ['attr_metric', 'attr_cnt1', 'attr_cnt2', 'attr_same_keys_proportion'], ATTRS_BERT_COSINE_VALUES_THRESHOLD,
            ATTRS_BERT_COSINE_KEYS_THRESHOLD
        ),
    )

    @property
    def config_feature_names(self) -> List[str]:
        feature_names = []
        for processor in self.feature_processors:
            feature_names.extend(processor.feature_names)
        return feature_names
