import pandas as pd
import json
from typing import List
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

from .feature_processor import FeatureProcessor
from matcher.utils.preprocess import append_color_to_attrs, parse_attributes
from matcher.models.matching_bert.scorer import BertScorer
from matcher.models.matching_bert.dataset import NamingMatchingDataset
from matcher.models.matching_bert.collator import Collator



class AttrMatchingBertFeature(FeatureProcessor):
    def __init__(self, feature_names: List[str], pretrained_model: str, needed_attrs_filename: str):
        super().__init__(feature_names)
        model = DebertaV2ForSequenceClassification.from_pretrained(pretrained_model)
        self.scorer = BertScorer(model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.batch_size = 32
        self.char_length_threshold = 200
        with open(needed_attrs_filename, "r") as f:
            self.needed_attrs = set(json.load(f))

    @property
    def processor_name(self) -> str:
        return "Matching BERT"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)

        def change_attrs_with_long_truncation(s):
            if s is not None:
                s = json.loads(s)
                if "Бренд" in s.keys():
                    s["Бренд"] = s["Бренд"][0].strip().lower()
                    if s["Бренд"] in ["нет бренда", ""]:
                        del s["Бренд"]
                if "Тип" in s.keys():
                    s["Тип"] = s["Тип"][0].strip().lower()
                if "Страна-изготовитель" in s.keys():
                    s["Страна-изготовитель"] = s["Страна-изготовитель"][0].strip().lower()
                if "Цвет товара" in s.keys():
                    del s["Цвет товара"]
                if "Название цвета" in s.keys():
                    del s["Название цвета"]
                if "Гарантийный срок" in s.keys():
                    s["Гарантийный срок"] = s["Гарантийный срок"][0].strip().lower()
                    if s["Гарантийный срок"] in garant_dict:
                        s["Гарантийный срок"] = garant_dict[s["Гарантийный срок"]]
                new_dict = dict()
                for key, item in s.items():
                    if key in self.needed_attrs:
                        if isinstance(item, str):
                            item = item.strip().lower()
                            new_dict[key] = item[:self.char_length_threshold]
                        elif isinstance(item, list):
                            key_string = []
                            for item_attr in item:
                                key_string.append(item_attr[:self.char_length_threshold].lower().strip())
                            if len(key_string) > 0:
                                new_dict[key] = " ".join(key_string)
                        else:
                            new_dict[key] = item
                return json.dumps(dict(sorted(new_dict.items(), key=lambda item: item[0])),
                                  ensure_ascii=False)
            return None

        garant_dict = {"12 мес": "1 год", "12": "1 год", "365": "1 год",
                       "12 месяцев": "1 год", "12 мес": "1 год",
                       "1 year": "1 год",
                       "30 дней": "1 месяц", "30": "1 месяц",
                       "31 день (с учетом сохранности товарного вида и упаковки)": "1 месяц",
                       "6 мес": "6 месяцев",
                       "6 month": "6 месяцев",
                       "24 мес.": "2 года",
                       "24 месяца": "2 года",
                       "36 мес.": "3 года",
                       "36 месяцев": "3 года",
                       "3 год": "3 года",
                       "14": "14 дней",
                       "14 дней на проверку": "14 дней", "2 недели": "14 дней",
                       "гарантия производителя": "официальная гарантия производителя"}
        df["new_attrs"] = df.characteristic_attributes_mapping.apply(change_attrs_with_long_truncation)
        df["new_attrs"] = df.apply(lambda x: append_color_to_attrs(x["new_attrs"], x["color_parsed"]), axis=1)
        df["attribute_string"] = df["new_attrs"].map(parse_attributes)
        return df

    def compute_pair_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df[["attribute_string1", "attribute_string2"]]
        dataset = NamingMatchingDataset(feature_df, self.tokenizer, "attribute_string1", "attribute_string2")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Collator(self.tokenizer), shuffle=False)
        predicted_probas = self.scorer.predict_proba(dataloader)
        df["attribute_matching_bert_score"] = predicted_probas
        return df