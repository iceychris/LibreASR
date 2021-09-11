import torch
from torch import nn
from torch.nn import functional as F


class Wav2Vec2Encoder(nn.Module):
    def __init__(
        self,
        feature_sz,
        hidden_sz,
        out_sz,
        dropout=0.01,
        dropout_input=0.0,
        dropout_inner=0.0,
        num_layers=2,
        trace=True,
        device="cuda:0",
        rnn_type="LSTM",
        norm="bn",
        attention=False,
        use_tmp_state_pcent=0.9,
        reversible=False,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.drop_input = nn.Dropout(dropout_input)

        # ff layer at end
        if not hidden_sz == out_sz:
            self.ff2 = nn.Linear(hidden_sz, out_sz)
        else:
            self.ff2 = nn.Sequential()

        # load wav2vec2 model
        from transformers import (
            AutoTokenizer,
            AutoModel,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Processor,
            Wav2Vec2Model,
        )

        name = kwargs.pop("wav2vec2_name", "facebook/wav2vec2-large-xlsr-53")
        cut_at = kwargs.pop("wav2vec2_cut", 15)
        model = Wav2Vec2Model.from_pretrained(name, gradient_checkpointing=True)
        model.encoder.layers = model.encoder.layers[:cut_at]
        self.wav2vec2 = model
        print("Wav2Vec2 model loaded:")
        print(f" => Using '{name}' model")
        print(f" => Keeping {cut_at} attention layers")

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward_wav2vec2(self, x, xl):
        # x must be raw audio of shape [B, T]
        if x.dim() == 4:
            x = x[:, :, 0, 0]
        elif x.dim() == 3:
            x = x[:, :, 0]

        # zero mean + unit variance
        if x.size(0) == 1:
            x = (x - x.mean()) / (x.std() + 1e-5)
        else:
            x = (x - x.mean(0)) / (x.std(0) + 1e-5)

        # attention mask / mask for later
        # TODO: this is maybe wrong...
        #       maybe use wav2vec2 tokenizer/processor?
        attn_mask = (
            torch.arange(x.size(1), dtype=xl.dtype, device=xl.device)[None, :]
            < xl[:, None]
        )

        # assemble
        inp = {"input_values": x, "attention_mask": attn_mask}

        # extract
        x = self.wav2vec2(**inp).last_hidden_state
        return x

    def forward(self, x, state=None, lengths=None, return_state=False):
        x = self.drop_input(x)

        # main block
        x = self.forward_wav2vec2(x, lengths)
        x = self.ff2(x)

        return x
