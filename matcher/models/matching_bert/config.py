from attr import frozen
from matcher.config import TEXT_DEBERTA_MATCHER


@frozen
class Config:
    data_dir: str = "."
    pretrained_model_name: str = "microsoft/mdeberta-v3-base"
    my_pretrained_model: str = TEXT_DEBERTA_MATCHER
    learning_rate: float = 1e-5
    batch_size: int = 8
    n_epoch: int = 3
    gradient_accumulation_steps: int = 8
    random_seed: int = 42
