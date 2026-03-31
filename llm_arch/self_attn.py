import torch.nn as nn
import torch

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, kv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.wq = nn.Linear(d_in, d_out, bias=kv_bias)
        self.wk = nn.Linear(d_in, d_out, bias=kv_bias)
        self.wv = nn.Linear(d_in, d_out, bias=kv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        ) 

    def forward(self,  x):
        b, num_tokens, d_in = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        attn_sc = q @ k.transpose(1, 2)
        attn_sc.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weight = torch.softmax(attn_sc / k.shape[-1] ** 0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)
        context_v = attn_weight @ v

        return context_v