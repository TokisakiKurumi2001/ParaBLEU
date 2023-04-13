import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from ReWord import ReWordModel, ReWordConfig
import evaluate
import numpy as np
import re
from transformers import RobertaTokenizer

class LitReWord(pl.LightningModule):
    def __init__(self, pretrained_ck: str, layers_use_from_last: int, method_for_layers: str, lr: float):
        super(LitReWord, self).__init__()
        config = ReWordConfig.from_pretrained(
            pretrained_ck,
            pretrained_ck=pretrained_ck,
            layers_use_from_last=layers_use_from_last,
            method_for_layers=method_for_layers)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_ck)
        self.tokenizer.add_tokens(['<ma>', '<madv>', '<mn>', '<mp>', '<ms>', '<mv>'])
        self.model = ReWordModel(config)
        self.vocab_size = self.model.config.vocab_size
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.valid_metric = evaluate.load("sacrebleu")
        self.test_metric = evaluate.load("sacrebleu")
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def __postprocess(self, preds, labels, eos_token_id=2):
        predictions = preds.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)
        loss = self.loss(logits.view(-1, self.vocab_size), labels.view(-1))
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)
        loss = self.loss(logits.view(-1, self.vocab_size), labels.view(-1))
        self.log('valid/loss', loss, sync_dist=True)

        preds = logits.argmax(dim=-1)
        decoded_preds, decoded_labels = self.__postprocess(preds, labels)
        self.valid_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def validation_epoch_end(self, outputs):
        results = self.valid_metric.compute()
        self.log('valid/sacre_bleu', results['score'], on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)
        loss = self.loss(logits.view(-1, self.vocab_size), labels.view(-1))
        self.log('test/loss', loss, sync_dist=True)

        preds = logits.argmax(dim=-1)
        decoded_preds, decoded_labels = self.__postprocess(preds, labels)
        self.test_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def test_epoch_end(self, outputs):
        results = self.test_metric.compute()
        self.log('test/sacre_bleu', results['score'], on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer