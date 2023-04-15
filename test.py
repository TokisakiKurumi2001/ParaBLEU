from ParaBLEU import ParaBLEUPDataLoader, ParaBLEUPretrainedModel
dataloader = ParaBLEUPDataLoader('xlm-roberta-base', 'facebook/m2m100_418M', 128)
[train_dataloader] = dataloader.get_dataloader(batch_size=2, types=['train'])
for batch in train_dataloader:
    # print(batch)
    break

model = ParaBLEUPretrainedModel('xlm-roberta-base', 'facebook/m2m100_418M')
gen_vocab_size = 128112
mlm_vocab_size = 250002
cls_class_num = 2
import torch.nn as nn
mlm_loss = nn.CrossEntropyLoss()
cls_loss = nn.CrossEntropyLoss()
gen_loss = nn.CrossEntropyLoss()

mlm_labels = batch.pop('mask_labels')
cls_labels = batch.pop('ent_labels')
gen_labels = batch.pop('gen_labels')
m, c, g = model(batch)

# breakpoint()
gen_loss = gen_loss(g.view(-1, gen_vocab_size), gen_labels.view(-1).long())
mlm_loss = mlm_loss(m.view(-1, mlm_vocab_size), mlm_labels.view(-1).long())
cls_loss = cls_loss(c.view(-1, cls_class_num), cls_labels.view(-1).long())

loss = gen_loss + 4.0 * mlm_loss + 10.0 * cls_loss
print(loss)

from transformers import M2M100Tokenizer
tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')

import numpy as np
def postprocess(preds, labels, eos_token_id=2):
    predictions = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels

import evaluate
gen_valid_metric = evaluate.load("metrics/sacrebleu.py")
cls_valid_metric = evaluate.load("metrics/accuracy.py")

preds = g.argmax(dim=-1)
decoded_preds, decoded_labels = postprocess(preds, gen_labels)
gen_valid_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

cls_preds = c.argmax(dim=-1)
cls_valid_metric.add_batch(predictions=cls_preds, references=cls_labels)

results = gen_valid_metric.compute()
print(results['score'])
results = cls_valid_metric.compute()
print(results['accuracy'])
