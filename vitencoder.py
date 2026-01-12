import math
from turtle import Turtle
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, image_size:int, patch_size:int, d_vision:int):
        super().__init__()
        assert image_size%patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_vision = d_vision


        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=d_vision,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        self.grid = image_size // patch_size
        self.n_patches = self.grid**2

    def forward(self, images: torch.Tensor)->torch.Tensor:
        B,C,H,W = images.shape
        assert C==3
        assert H==self.image_size and W==self.image_size

        x = self.proj(images)
        B,D,Gh,Gw = x.shape
        assert Gh == self.grid and Gw == self.grid

        x = x.flatten(2).transpose(1,2)
        assert x.shape==(B, self.n_patches, self.d_vision)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads:int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3*d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self,  x:torch.Tensor)->torch.Tensor:
        B,T,D = x.shape
        assert D == self.d_model
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3,dim=-1)

        q = q.view(B,T, self.n_heads, self.d_heads).transpose(1,2)
        k = k.view(B,T, self.n_heads, self.d_heads).transpose(1,2)
        v = v.view(B,T, self.n_heads, self.d_heads).transpose(1,2)


        att = (q@k.transpose(-2,-1))/math.sqrt(self.d_heads)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att@v
        y = y.transpose(1,2).contiguous().view(B,T,D)
        y = self.out(y)
        return y

class MLP(nn.Module):
    def __init__(self, d_model:int, d_ff: int, dropout: float=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViTVisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.patch = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.d_vision)
        assert self.patch.n_patches == cfg.n_patches

        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.d_vision))
        self.drop = nn.Dropout(0.0)

        d_ff = 4*cfg.d_vision
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(cfg.d_vision, cfg.n_heads, d_ff, dropout=0.0) for _ in range(cfg.n_layers)
            ])
        self.ln_out = nn.LayerNorm(cfg.d_vision)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)


    def forward(self, images: torch.Tensor)->torch.Tensor:
        x = self.patch(images)
        x = x + self.pos_emb
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x)
        return x

def phase1_smoke_test(device="cpu"):
    torch.manual_seed(0)

    class LVLMConfig:
        def __init__(
            self,
            image_size=32,
            patch_size=32,
            d_vision=128,
            d_model=128,
            n_heads=4,
            n_layers=2,
            vocab_size=200,
            max_text_len=16,
            n_image_tokens=None
        ):
            assert image_size % patch_size==0
            self.image_size = image_size
            self.patch_size = patch_size
            self.d_vision = d_vision
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.vocab_size = vocab_size
            self.max_text_len = max_text_len
            self.n_image_tokens = n_image_tokens
            patches_per_side = image_size // patch_size
            self.n_patches = patches_per_side ** 2
            self.n_image_tokens = n_image_tokens or self.n_patches

        
    cfg = LVLMConfig()
    vision = ViTVisionEncoder(cfg).to(device)
    images = torch.randn(2, 3, cfg.image_size, cfg.image_size, device=device)

    image_tokens = vision(images)
    print("image_tokens: ", image_tokens.shape)
    assert image_tokens.shape == (2, cfg.n_patches, cfg.d_vision)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phase1_smoke_test(device=device)












    


