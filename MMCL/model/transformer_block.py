import copy
import torch.nn as nn
from .block import Block
from torch.nn import LayerNorm

class TransformerBlock(nn.Module):

    def __init__(self, dim, intermediate_dim, num_labels, num_heads= 8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(2):
            layer = Block(dim, intermediate_dim, num_labels)
            self.layer.append(copy.deepcopy(layer))


    def forward(self, x, k, v):
        for layer_block in self.layer:
            x= layer_block(x, k, v)
        encoded = self.encoder_norm(x)
        return encoded