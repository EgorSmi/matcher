from transformers import AutoTokenizer
import torch


class Collator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.max_input_length = 512

    def __call__(self, batch):
        input_encoding = self.tokenizer(
            batch,
            padding='longest',
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        token_type_ids_idx = torch.arange(0, attention_mask.shape[1])
        salt_raw = torch.tensor([self.tokenizer.sep_token_id] * attention_mask.shape[0]).view(-1, 1)
        salt_input_ids = torch.concat([input_ids, salt_raw], axis=1)
        sep_idx = torch.argmax((salt_input_ids == self.tokenizer.sep_token_id).to(dtype=torch.int), dim=-1).view(-1, 1)
        condition = token_type_ids_idx <= sep_idx
        token_type_ids = torch.where(condition, 0, 1)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }
