import pandas as pd
from typing import List
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, XLMRobertaForSequenceClassification

from .feature_processor import FeatureProcessor
from matcher.utils.preprocess import preprocess
from matcher.models.matching_bert.scorer import BertScorer
from matcher.models.matching_bert.dataset import NamingMatchingDataset
from matcher.models.matching_bert.collator import Collator



class MatchingBertFeature(FeatureProcessor):
    def __init__(self, feature_names: List[str], pretrained_model: str, with_type_ids: bool = True):
        super().__init__(feature_names)
        if with_type_ids:
            model = DebertaV2ForSequenceClassification.from_pretrained(pretrained_model)
        else:
            model = XLMRobertaForSequenceClassification.from_pretrained(pretrained_model)
        self.scorer = BertScorer(model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.batch_size = 32
        self.with_type_ids = with_type_ids

    @property
    def processor_name(self) -> str:
        return "Matching BERT"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        if "name_for_bert" not in df.columns:
            df["name_for_bert"] = df["name"].map(preprocess)
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df[["name_for_bert1", "name_for_bert2"]]
        dataset = NamingMatchingDataset(feature_df, self.tokenizer, "name_for_bert1", "name_for_bert2")
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=Collator(self.tokenizer, self.with_type_ids), shuffle=False
        )
        predicted_probas = self.scorer.predict_proba(dataloader)
        df[self.feature_names[0]] = predicted_probas
        return df
