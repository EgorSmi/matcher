from dataclasses import dataclass
from attr import frozen
from typing import Tuple, List
from .feature_processor import FeatureProcessor
from matcher.config import TEXT_DEBERTA_MATCHER


@frozen
class Config:
    data_dir: str
    text_matcher_filepath: str = TEXT_DEBERTA_MATCHER
    feature_processors: Tuple[FeatureProcessor]
