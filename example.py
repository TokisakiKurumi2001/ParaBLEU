from ReWord import ReWordModel
from transformers import RobertaTokenizer
import torch
import time
pretrained_ck = 'reword_model/v1'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_ck, add_prefix_space=True)
model = ReWordModel.from_pretrained(pretrained_ck)
model.eval()
sent = "I education company . <ma> <mp> <mv>"
inputs = [sent.split(" ")]
tokenized_inputs = tokenizer(
    inputs,
    padding="max_length",
    truncation=True,
    max_length=25,
    is_split_into_words=True,
    return_tensors="pt"
)
print(tokenized_inputs)
with torch.no_grad():
    start_time = time.time()
    logits = model(**tokenized_inputs)
    print("--- %s seconds ---" % (time.time() - start_time))

preds = logits.argmax(dim=-1)
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
print(decoded_preds)
