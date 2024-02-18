import torch.nn as nn
from torch.nn import LayerNorm
from .attention import Attention
from .mlp import Mlp

class Block(nn.Module):
    def __init__(self, dim, intermediate_dim, num_labels, num_heads= 24, re=0.5):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.intermediate_dim = intermediate_dim
        self.num_labels = num_labels

        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)

        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)

        self.ffn = Mlp(self.hidden_size, self.intermediate_dim, self.num_labels)

        self.attn = Attention(dim, num_heads, re)

    def forward(self, x, k, v):
        h = x
        x = self.attention_norm(x)
        x= self.attn(x, k, v)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x