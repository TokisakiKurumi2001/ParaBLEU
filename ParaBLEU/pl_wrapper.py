import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from ParaBLEU import ParaBLEUPretrainedModel
import evaluate
import numpy as np
import re
from transformers import M2M100Tokenizer, get_constant_schedule_with_warmup

class LitParaBLEU(pl.LightningModule):
    def __init__(self, encoder_ckpt: str, gen_ckpt: str, lr: float, alpha: float, beta: float):
        super(LitParaBLEU, self).__init__()
        self.tokenizer = M2M100Tokenizer.from_pretrained(gen_ckpt)
        self.model = ParaBLEUPretrainedModel('xlm-roberta-base', 'facebook/m2m100_418M')
        self.gen_vocab_size = self.model.gen_vocab_size
        self.mlm_vocab_size = self.model.mlm_vocab_size
        self.cls_class_num = self.model.config.cls_out
        self.mlm_loss = nn.CrossEntropyLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.gen_loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gen_valid_metric = evaluate.load("metrics/sacrebleu.py")
        self.gen_test_metric = evaluate.load("metrics/sacrebleu.py")
        self.cls_valid_metric = evaluate.load("metrics/accuracy.py")
        self.cls_test_metric = evaluate.load("metrics/accuracy.py")
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

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
        mlm_labels = batch.pop('mask_labels')
        cls_labels = batch.pop('ent_labels')
        gen_labels = batch.pop('gen_labels')
        m, c, g = self.model(batch)

        gen_loss = self.gen_loss(g.view(-1, self.gen_vocab_size), gen_labels.view(-1).long())
        mlm_loss = self.mlm_loss(m.view(-1, self.mlm_vocab_size), mlm_labels.view(-1).long())
        cls_loss = self.cls_loss(c.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        loss = gen_loss + self.alpha * mlm_loss + self.beta * cls_loss
        self.log("train/gen_loss", gen_loss, sync_dist=True)
        self.log("train/mlm_loss", mlm_loss, sync_dist=True)
        self.log("train/cls_loss", cls_loss, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mlm_labels = batch.pop('mask_labels')
        cls_labels = batch.pop('ent_labels')
        gen_labels = batch.pop('gen_labels')
        m, c, g = self.model(batch)

        gen_loss = self.gen_loss(g.view(-1, self.gen_vocab_size), gen_labels.view(-1).long())
        mlm_loss = self.mlm_loss(m.view(-1, self.mlm_vocab_size), mlm_labels.view(-1).long())
        cls_loss = self.cls_loss(c.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        loss = gen_loss + self.alpha * mlm_loss + self.beta * cls_loss
        self.log("valid/gen_loss", gen_loss, sync_dist=True)
        self.log("valid/mlm_loss", mlm_loss, sync_dist=True)
        self.log("valid/cls_loss", cls_loss, sync_dist=True)
        self.log("valid/loss", loss, sync_dist=True)

        preds = g.argmax(dim=-1)
        decoded_preds, decoded_labels = self.__postprocess(preds, gen_labels)
        self.gen_valid_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        cls_preds = c.argmax(dim=-1)
        self.cls_valid_metric.add_batch(predictions=cls_preds, references=cls_labels)

    def validation_epoch_end(self, outputs):
        results = self.gen_valid_metric.compute()
        self.log('valid/sacre_bleu', results['score'], on_epoch=True, on_step=False, sync_dist=True)
        results = self.cls_valid_metric.compute()
        self.log('valid/accuracy', results['accuracy'], on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        mlm_labels = batch.pop('mask_labels')
        cls_labels = batch.pop('ent_labels')
        gen_labels = batch.pop('gen_labels')
        m, c, g = self.model(batch)

        gen_loss = self.gen_loss(g.view(-1, self.gen_vocab_size), gen_labels.view(-1).long())
        mlm_loss = self.mlm_loss(m.view(-1, self.mlm_vocab_size), mlm_labels.view(-1).long())
        cls_loss = self.cls_loss(c.view(-1, self.cls_class_num), cls_labels.view(-1).long())

        loss = gen_loss + self.alpha * mlm_loss + self.beta * cls_loss
        self.log("test/gen_loss", gen_loss, sync_dist=True)
        self.log("test/mlm_loss", mlm_loss, sync_dist=True)
        self.log("test/cls_loss", cls_loss, sync_dist=True)
        self.log("test/loss", loss, sync_dist=True)

        preds = g.argmax(dim=-1)
        decoded_preds, decoded_labels = self.__postprocess(preds, gen_labels)
        self.gen_test_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        cls_preds = c.argmax(dim=-1)
        self.cls_test_metric.add_batch(predictions=cls_preds, references=cls_labels)

    def test_epoch_end(self, outputs):
        results = self.gen_test_metric.compute()
        self.log('test/sacre_bleu', results['score'], on_epoch=True, on_step=False, sync_dist=True)
        results = self.cls_valid_metric.compute()
        self.log('test/accuracy', results['accuracy'], on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, 1500)
        return [optimizer], [lr_scheduler]
