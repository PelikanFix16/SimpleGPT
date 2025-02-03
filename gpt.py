import torch
from torch import nn
import torch.nn.functional as F
from decoder import DecoderLayers


class Gpt(nn.Module):
    def __init__(
        self, vocab_size, d_model, ff_layers, num_heads, seq_len, dropout, num_layers
    ):
        super(Gpt, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(seq_len, d_model)
        self.decoder = DecoderLayers(d_model, ff_layers, num_heads, dropout, num_layers)
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        bsz, seq_len = x.size()
        x_emb = self.embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)
        x_emb = x_emb + pos_emb
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)
        decoder_out = self.decoder(x_emb, causal_mask)
        out = self.linear_out(decoder_out)
        return out

    def generate_causal_mask(self, seq_len):

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

        mask = torch.triu(mask, diagonal=1)
        return mask
