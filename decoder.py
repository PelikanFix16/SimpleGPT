import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, d_model, ff_layers, num_heads, dropout):
        super(Decoder, self).__init__()
        self.multi_head = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ff1 = nn.Linear(d_model, ff_layers)
        self.ff2 = nn.Linear(ff_layers, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_t = x.transpose(0, 1)
        attn_output, attn_weights = self.multi_head(x_t, x_t, x_t, attn_mask=mask)
        attention = attn_output.transpose(0, 1)
        x = self.norm1(x + self.dropout1(attention))
        ff_out = self.ff2(F.relu(self.ff1(x)))
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class DecoderLayers(nn.Module):
    def __init__(self, d_model, ff_layers, num_heads, dropout, num_layers):
        super(DecoderLayers, self).__init__()
        self.layers = nn.ModuleList(
            [Decoder(d_model, ff_layers, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
