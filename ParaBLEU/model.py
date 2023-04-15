from transformers import PreTrainedModel, XLMRobertaModel, M2M100ForConditionalGeneration
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Optional, Tuple, Dict
from ParaBLEU import ParaMapConfig

class ParaMapPreTrainedModel(PreTrainedModel):
    config_class = ParaMapConfig
    base_model_prefix = "para_map"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class ParaMapModel(ParaMapPreTrainedModel):
    def __init__(self, config: ParaMapConfig):
        super().__init__(config)
        self.config = config
        
        # MASK prediction
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        if config.act_fn == "gelu":
            self.mlm_act_fn = nn.GELU()
        elif config.act_fn == "relu":
            self.mlm_act_fn = nn.ReLU()
        else:
            self.mlm_act_fn = nn.Identity()
        self.mlm_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlm_out = nn.Linear(config.hidden_size, config.mlm_vocab_size)

        # Entailment classification
        self.cls_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.cls_act_fn = nn.Tanh()
        self.cls_dropout = nn.Dropout(config.drop_prob)
        self.cls_out = nn.Linear(config.hidden_size, config.cls_out)

        # FFN
        self.ffn_down = nn.Linear(config.hidden_size, config.ffn_hidden_size)
        if config.act_fn == "gelu":
            self.ffn_act_fn = nn.GELU()
        elif config.act_fn == "relu":
            self.ffn_act_fn = nn.ReLU()
        else:
            self.ffn_act_fn = nn.Identity()
        self.ffn_up = nn.Linear(config.ffn_hidden_size, config.gen_hidden_size)

        self.post_init()

    def forward(self, input_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Tensor:
        # receive the output embedding of the encoder model
        
        # mask prediction
        m = self.mlm_act_fn(self.mlm_dense(input_embeddings))
        m = self.mlm_out(self.mlm_ln(m))

        # classification
        # mean pooling
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(input_embeddings.size()).float()
        sum_embeddings = torch.sum(input_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vectors.append(sum_embeddings / sum_mask)
        sentence_vector = torch.cat(output_vectors, 1)

        c = self.cls_act_fn(self.cls_dense(sentence_vector))
        c = self.cls_out(self.cls_dropout(c))

        # ffn
        f = self.ffn_up(self.ffn_act_fn(self.ffn_down(sentence_vector)))

        return m, c, f

class ParaBLEUPretrainedModel(nn.Module):
    def __init__(self, encoder_ckpt: str, gen_ckpt: str, mapper_ckpt: str = '', mode: str="train"):
        super(ParaBLEUPretrainedModel, self).__init__()
        if mode == "train":
            config = ParaMapConfig()
            self.mapper = ParaMapModel(config)
        else:
            self.mapper = ParaMapModel.from_pretrained(mapper_ckpt)

        self.encoder = XLMRobertaModel.from_pretrained(encoder_ckpt)
        self.generator = M2M100ForConditionalGeneration.from_pretrained(gen_ckpt)

        self.norm = nn.LayerNorm(config.gen_hidden_size, eps=config.layer_norm_eps)

        self.mlm_vocab_size = config.mlm_vocab_size
        self.gen_vocab_size = 128112

    def save_pretrained(self, path):
        self.mapper.save_pretrained(path + "/mapper")
        self.encoder.save_pretrained(path + "/encoder")
        self.generator.save_pretrained(path + "/generator")

    def forward(self, inputs):
        enc_out = self.encoder(input_ids=inputs['mask_input_ids'], attention_mask=inputs['mask_attention_mask'])
        m, c, f = self.mapper(input_embeddings=enc_out.last_hidden_state, attention_mask=inputs['mask_attention_mask'])
        gen_enc_out = self.generator.model.encoder(input_ids=inputs['gen_input_ids'], attention_mask=inputs['gen_attention_mask'])

        # up sample f to match sequence length
        batch_size, seq_len, dimension = gen_enc_out.last_hidden_state.shape
        f_up = f.repeat(1, seq_len).reshape(batch_size, seq_len, dimension)
        combine_gen_out = self.norm(f_up + gen_enc_out.last_hidden_state)
        gen_dec_out = self.generator.model.decoder(
            input_ids=inputs['gen_decoder_input_ids'],
            encoder_attention_mask=inputs['gen_attention_mask'],
            encoder_hidden_states=combine_gen_out
        )
        gen_out = self.generator.lm_head(gen_dec_out.last_hidden_state)
        return m, c, gen_out
