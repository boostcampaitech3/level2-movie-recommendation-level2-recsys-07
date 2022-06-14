from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks, get_best_10

import torch
import torch.nn as nn


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, inference_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, inference_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
    
    def inference_items(self, batch, user_seen):
        seqs = batch[0]
        seqs = seqs.unsqueeze(0)
        scores = self.model(seqs)
        scores = scores[:, -1, :]
        best10 = get_best_10(scores, user_seen)
        return best10
