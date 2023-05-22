import pandas as pd
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

from .feature_processor import FeatureProcessor
from matcher.utils.preprocess import preprocess
from matcher.models.matching_bert.scorer import BertScorer
from matcher.models.matching_bert.dataset import NamingMatchingDataset
from matcher.models.matching_bert.collator import Collator
from matcher.models.matching_bert.config import Config



class MatchingBertFeature(FeatureProcessor):
    def __init__(self, pretrained_model: str):
        super().__init__()
        model = DebertaV2ForSequenceClassification.from_pretrained(pretrained_model)
        self.scorer = BertScorer(model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    @property
    def processor_name(self) -> str:
        return "Matching BERT"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        df["name"] = df["name"].map(preprocess)
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df[["name1", "name2"]]
        dataset = NamingMatchingDataset(df, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=Config.batch_size, collate_fn=Collator(self.tokenizer), shuffle=False)
        predicted_probas = self.scorer.predict_proba(dataloader)
        df["matching_bert_score"] = predicted_probas
        return df
