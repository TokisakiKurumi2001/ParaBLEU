# Word ordering

Sequence tagging to predict the correct order of input sequence trained on dedicated dataset

## How to use

```bash
pip install ReWord
```

```python
from ReWord import ReWordModel
from transformers import RobertaTokenizer
import torch
pretrained_ck = 'transZ/reword'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_ck, add_prefix_space=True)
model = ReWordModel.from_pretrained(pretrained_ck)
model.eval()
sent = "I education company . <ma> <mp> <mv>"
inputs = sent.split(" ")
tokenized_inputs = tokenizer(
    inputs,
    padding="max_length",
    truncation=True,
    max_length=25,
    is_split_into_words=True,
    return_tensors="pt"
)
with torch.no_grad():
    logits = model(**tokenized_inputs)

preds = logits.argmax(dim=-1)
decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
print(decoded_preds) # "I <mv> <mp> <ma> education company ."
```