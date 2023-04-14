from transformers.configuration_utils import PretrainedConfig

class ParaMapConfig(PretrainedConfig):
    model_type = "para_map"

    def __init__(
        self,
        mlm_vocab_size=250002,
        hidden_size=768,
        cls_out=2,
        ffn_hidden_size=64,
        act_fn="relu",
        gen_hidden_size=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        drop_prob=0.1,
        **kwargs,
    ):
        super().__init__(pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)
        self.mlm_vocab_size = mlm_vocab_size
        self.hidden_size = hidden_size
        self.cls_out = cls_out
        self.ffn_hidden_size = ffn_hidden_size
        self.act_fn = act_fn
        self.gen_hidden_size = gen_hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.drop_prob = drop_prob