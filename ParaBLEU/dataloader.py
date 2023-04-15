from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple
from transformers import XLMRobertaTokenizerFast, M2M100Tokenizer
import random
from copy import deepcopy

class ParaBLEUPDataLoader:
    def __init__(self, encoder_ckpt: str, s2s_ckpt: str, max_length: int):
        dataset = load_dataset('csv', data_files=["data/paws-x-parableu.csv"])
        dataset = dataset['train'].train_test_split(test_size=0.05, seed=42)
        self.encoder_tokenizer = XLMRobertaTokenizerFast.from_pretrained(encoder_ckpt)
        self.s2s_tokenizer = M2M100Tokenizer.from_pretrained(s2s_ckpt)
        self.max_length = max_length
        random.seed(42)
        self.dataset = dataset.map(
            self.__tokenize,
            remove_columns=dataset["train"].column_names
        )

    def __shift_tokens_left(self, input_ids: torch.Tensor, pad_token_id: int):
        shifted_input_ids = torch.zeros(input_ids.shape, dtype=torch.int32)
        shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
        shifted_input_ids[:, -1] = pad_token_id

        return shifted_input_ids

    def __tokenize_gen(self, text1: str, text2: str, lang_code: str):
        self.s2s_tokenizer.src_lang = lang_code
        self.s2s_tokenizer.tgt_lang = lang_code
        inps = self.s2s_tokenizer(text1, text_target=text2, truncation=True, padding="max_length", max_length=self.max_length)
        inps['decoder_input_ids'] = inps.pop('labels')
        return inps

    def __tokenize_mlm(self, text1: str, text2: str, threshold: float=0.15):
        inps = self.encoder_tokenizer(text1, text2, truncation=True, padding="max_length", max_length=self.max_length)
        activate = False
        labels = []
        for i, tok in enumerate(inps['input_ids']):
            if tok == 2:
                activate = True
                labels.append(-100)
                continue
            if tok == 1:
                labels.append(-100)
                continue
            if activate:
                if random.random() < threshold:
                    labels.append(inps['input_ids'][i])
                    inps['input_ids'][i] = self.encoder_tokenizer.mask_token_id
                    continue
            labels.append(-100)

        inps['labels'] = labels
        return inps

    def __tokenize(self, examples):
        mask = self.__tokenize_mlm(examples['sentence1'], examples['sentence2'])
        gen = self.__tokenize_gen(examples['sentence1'], examples['sentence2'], examples['lang'])
        
        rt_dict = {}
        for k, v in mask.items():
            rt_dict[f'mask_{k}'] = v
        for k, v in gen.items():
            rt_dict[f'gen_{k}'] = v
        rt_dict['ent_labels'] = examples['label']
        return rt_dict

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples
        # mask_toks = [self.__tokenize_mlm(inp['sentence1'], inp['sentence2']) for inp in examples]
        # mask_toks = {key: [example[key] for example in mask_toks] for key in mask_toks[0].keys()}

        # batch = {k: torch.tensor(v, dtype=torch.int32) for k, v in mask_toks.items()}
        # batch['ent_labels'] = torch.tensor(encoded_inputs['label'], dtype=torch.int32)

        # gen_toks = [self.__tokenize_gen(inp['sentence1'], inp['sentence2'], "en") for inp in examples]
        # gen_toks = {key: [example[key] for example in gen_toks] for key in gen_toks[0].keys()} 
        # gen_toks = {k: torch.tensor(v, dtype=torch.int32) for k, v in gen_toks.items()}
        
        # convert list to tensor
        batch = {k: torch.tensor(v, dtype=torch.int32) for k, v in encoded_inputs.items()}
        batch['gen_labels'] = self.__shift_tokens_left(batch['gen_decoder_input_ids'], self.s2s_tokenizer.pad_token_id)

        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "test"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=1)
            )
        return res
