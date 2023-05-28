from dataclasses import dataclass
from typing import Tuple, List
from .feature_processor import FeatureProcessor
from .matching_bert_feature import MatchingBertFeature
from .simple_categories_features import SimpleCategoriesFeatures
from .pics_cosine_features import PicsCosineFeatures
from .color_features import SameColorFeatures
from .bert64_embedding_features import Bert64EmbeddingFeatures
from .resnet128_embedding_features import Resnet128EmbeddingFeatures
from .matching_attribute_bert import AttrMatchingBertFeature
from matcher.config import (
    TEXT_DEBERTA_MATCHER, PICS_COSINE_SIMILARITY_THRESHOLD, ATTRIBUTE_DEBERTA_MATCHER, NEEDED_ATTRS,
    XLM_ROBERTA_NAME_MATCHER,
)


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
            ["pics_norm_sum", "pics_min"], PICS_COSINE_SIMILARITY_THRESHOLD
        ),
        SameColorFeatures(
            ["same_colorname", "same_hex", "same_rgb", "color_difference_ciede"]
        ),
        AttrMatchingBertFeature(
            ["attribute_matching_bert_score"], ATTRIBUTE_DEBERTA_MATCHER, NEEDED_ATTRS,
        ),
        MatchingBertFeature(["name_matching_xlmroberta_score"], XLM_ROBERTA_NAME_MATCHER, False)
    )
    embedding_features_processors: Tuple[FeatureProcessor] = (
        Bert64EmbeddingFeatures(
            ["name_bert_641", "name_bert_642"]
        ),
        Resnet128EmbeddingFeatures(
            ["main_pic_embeddings_resnet_v11", "main_pic_embeddings_resnet_v12"]
        )
    )

    @property
    def config_feature_names(self) -> List[str]:
        feature_names = []
        for processor in self.feature_processors:
            feature_names.extend(processor.feature_names)
        feature_names.extend(self.config_embedding_features)
        return feature_names

    @property
    def config_embedding_features(self) -> List[str]:
        feature_names = []
        for processor in self.embedding_features_processors:
            feature_names.extend(processor.feature_names)
        return feature_names
