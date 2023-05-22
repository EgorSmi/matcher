from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import logging


LOG = logging.getLogger(__name__)


class BertScorer:
    def __init__(self, model: nn.Module):
        self.accelerator = Accelerator()
        self.model = model
        self.model.eval()
        self.model.to(self.accelerator.device)
        LOG.debug(f"Running on {self.accelerator.device}..")

    def _batch_processing(self, batch):
        outputs = self.model(**batch)
        logits = outputs.logits
        return logits

    @torch.no_grad()
    def predict_proba(self, dataloader: DataLoader) -> list:
        dataloader = self.accelerator.prepare_data_loader(dataloader)
        probas = []
        with tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                logits = self._batch_processing(batch)
                proba = nn.functional.softmax(logits, dim=-1)[:, 1]
                probas.extend(proba.cpu().numpy().tolist())
                torch.cuda.empty_cache()
                pbar.update(1)
        return probas
