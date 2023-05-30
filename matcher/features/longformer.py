import pandas as pd
from typing import List

from .feature_processor import FeatureProcessor
import pandas as pd

import torch
from tqdm import tqdm
import numpy as np
import torch.utils.data.distributed

from transformers import (
    LongformerTokenizerFast, AutoTokenizer
)
from transformers.models.longformer.modeling_longformer import LongformerForSequenceClassification
from matcher.config import (
    LONGFORMER_BATCHSIZE, PATH_TO_LONGFORMER
)



class LongformerFeature(FeatureProcessor):
    def __init__(self, feature_names: List[str], pretrained_model: str, needed_attrs_filename: str):
        super().__init__(feature_names)
        self.auto_tokenizer = AutoTokenizer.from_pretrained(PATH_TO_LONGFORMER)
        self.auto_tokenizer.add_tokens(["<eoi>"], special_tokens=True)
        self.auto_tokenizer.eoi_token = "<eoi>"
        self.longformer_xlm_roberta = LongformerForSequenceClassification.from_pretrained(PATH_TO_LONGFORMER).to('cuda')
        self.longformer_xlm_roberta.config.problem_type = "single_label_classification"
        self.longformer_xlm_roberta.resize_token_embeddings(len(self.auto_tokenizer))

    @property
    def processor_name(self) -> str:
        return "Matching longformer"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        return df

    def compute_pair_feature(self, train_df: pd.DataFrame) -> pd.DataFrame:
        train_df_data = []
        for i in range(len(train_df)):
            name1 = train_df.loc[i, "name1"]
            name2 = train_df.loc[i, "name2"]
            attr1 = train_df.loc[i, "attribute_string1"]
            attr2 = train_df.loc[i, "attribute_string2"]
            text_input = (
                    self.auto_tokenizer.bos_token + name1 + " " + attr1 + self.auto_tokenizer.sep_token
                    + name2 + " " + attr2 + self.auto_tokenizer.eos_token
            )
            train_df_data.append(text_input)
        with torch.no_grad():
            bs = LONGFORMER_BATCHSIZE
            logits = []
            for dfg in tqdm(range(len(train_df_data) // bs)):
                logits.append(torch.nn.functional.softmax(self.longformer_xlm_roberta(**auto_tokenizer(
                    train_df_data[dfg * bs:(dfg + 1) * bs],
                    padding='longest',
                    max_length=2048,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to('cuda')).logits, dim=-1)[:, 1].cpu().numpy())
            if len(train_df_data) % bs != 0:
                logits.append(torch.nn.functional.softmax(self.longformer_xlm_roberta(**self.auto_tokenizer(
                    train_df_data[len(train_df_data) - len(train_df_data) % bs:],
                    padding='longest',
                    # padding='max_length',
                    max_length=2048,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to('cuda')).logits, dim=-1)[:, 1].cpu().numpy())
        train_df['longformer'] = pd.Series(np.concatenate((np.array(logits[:-1]).flatten(), logits[-1])))
        return train_df
