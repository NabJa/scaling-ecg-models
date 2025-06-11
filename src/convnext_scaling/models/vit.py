from typing import Optional

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn


def exists(val):
    return val is not None


def posemb_sincos_1d(patches, temperature=10000, dtype=torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device=device)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        rotary_embed: Optional[RotaryEmbedding] = None,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.rotary_embed = rotary_embed

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_mult, rotary_embed, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            rotary_embed=rotary_embed,
                            dropout=dropout,
                        ),
                        FeedForward(dim, mlp_mult, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ECGViT(nn.Module):
    def __init__(
        self,
        *,
        seq_len,
        patch_size,
        num_classes=26,
        dim=128,
        depth=2,
        heads=8,
        mlp_mult=4,
        channels=12,
        dim_head=32,
        rotary_embed=False,
        dropout=0.0,
    ):
        """
        Args:
            seq_len: length of input sequence.
            patch_size: size of the patch.
            num_classes: number of classes.
            dim: dimension of the model.
            depth: depth of the model.
            heads: number of heads.
            mlp_mult: multiplier for the mlp dimension.
            channels: number of channels.
            dim_head: dimension of the head.
            rotary_embed: whether to use rotary embeddings. If False, sinusoidal embeddings are used.
            dropout: dropout rate.
        """
        super().__init__()

        assert seq_len % patch_size == 0

        patch_dim = channels * patch_size

        self.rotary_embed = RotaryEmbedding(dim_head) if rotary_embed else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) -> b n (p c)", p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_mult, self.rotary_embed, dropout=dropout
        )

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, series):
        x = self.to_patch_embedding(series)

        # Add class token
        b, _, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        if not exists(self.rotary_embed):
            x = x + posemb_sincos_1d(x)

        x = self.transformer(x)

        return self.linear_head(x[:, 0])
