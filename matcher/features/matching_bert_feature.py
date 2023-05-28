import pandas as pd
from typing import List
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

from .feature_processor import FeatureProcessor
from matcher.utils.preprocess import preprocess
from matcher.models.matching_bert.scorer import BertScorer
from matcher.models.matching_bert.dataset import NamingMatchingDataset
from matcher.models.matching_bert.collator import Collator



class MatchingBertFeature(FeatureProcessor):
    def __init__(self, feature_names: List[str], pretrained_model: str):
        super().__init__(feature_names)
        model = DebertaV2ForSequenceClassification.from_pretrained(pretrained_model)
        self.scorer = BertScorer(model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.batch_size = 32

    @property
    def processor_name(self) -> str:
        return "Matching BERT"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        df["name_for_bert"] = df["name"].map(preprocess)
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df[["name_for_bert1", "name_for_bert2"]]
        dataset = NamingMatchingDataset(feature_df, self.tokenizer, "name_for_bert1", "name_for_bert2")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Collator(self.tokenizer), shuffle=False)
        predicted_probas = self.scorer.predict_proba(dataloader)
        df["matching_bert_score"] = predicted_probas
        return df
