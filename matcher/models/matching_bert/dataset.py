from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class NamingMatchingDataset(Dataset):
    def __init__(self, text_df: pd.DataFrame, tokenizer: AutoTokenizer):
        self.name1 = text_df["name1"].tolist()
        self.name2 = text_df["name2"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.name1)

    def __getitem__(self, idx: int):
        naming_1 = self.name1[idx]
        naming_2 = self.name2[idx]
        text_input = self.tokenizer.cls_token + naming_1 + self.tokenizer.sep_token + naming_2 + self.tokenizer.eos_token
        return text_input
