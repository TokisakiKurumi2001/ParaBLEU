from ReWord import ReWordModel
from transformers import RobertaTokenizer
import torch
import time
# miss_sentence_component = MissSentComp('misecom_model/v1', 'roberta-base')
# sent = "I education company."
# print(miss_sentence_component(sent))
pretrained_ck = 'reword_model/v1'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_ck)#, add_prefix_space=True)
model = ReWordModel.from_pretrained(pretrained_ck)
tokenizer.push_to_hub('transZ/reword')
model.push_to_hub('transZ/reword')