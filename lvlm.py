import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LVLMConfig:
    def __init__(
        self,
        image_size = 32,
        patch_size = 8,
        d_vision = 128,
        d_model = 128,
        n_heads = 4,
        n_layers = 4,
        vocab_size = 200,
        max_text_len = 32,
        n_image_tokens = None

    ):
        assert image_size % patch_size == 0, "image_size must be divisible by path_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_vision = d_vision
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len

        patches_per_side = image_size // patch_size
        self.n_patches = patches_per_side*patches_per_side
        self.n_image_tokens = n_image_tokens or self.n_patches



class VisionEncoder(nn.Module):
    def __init__(self,cfg:LVLMConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(3, cfg.d_vision)


    def forward(self, images : torch.Tensor)-> torch.Tensor:
        B, C, H, W = images.shape
        assert C == 3, "Expecting 3 channels (RGB)"
        assert H == self.cfg.image_size and W == self.cfg.image_size, "Image dimensions must match configuration"
        
        T_img = self.cfg.n_patches
        x = torch.randn(B, T_img, 3, device =images.device, dtype=images.dtype)
        return self.proj(x)


class TextDecoder(nn.Module):
    def __init__(self,cfg:LVLMConfig):
        super().__init__()
        self.cfg = cfg
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        B,T,D = x.shape
        assert D == self.cfg.d_model, "Input embedding dimension must match model dimension"
        return self.lm_head(x)

class MiniLVLM (nn.Module):
    def __init__(self, cfg:LVLMConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg)
        self.projector = nn.Linear(cfg.d_vision, cfg.d_model)

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_text_len + cfg.n_image_tokens, cfg.d_model)
        self.decode = TextDecoder(cfg)

    def forward(self, images: torch.Tensor, input_ids:torch.Tensor)-> torch.Tensor:
        B, T_txt = input_ids.shape
        assert T_txt <= self.cfg.max_text_len

        img_tokens = self.vision(images)
        img_tokens = self.projector(img_tokens)

        txt_tokens = self.token_emb(input_ids)

        x = torch.cat([img_tokens, txt_tokens], dim=1)
        T_total = x.size(1)

        pos = torch.arange(T_total, device = x.device).unsqueeze(0).expand(B, T_total)
        x = x + self.pos_emb(pos)
        logits = self.decode(x)
        return logits


def phase0_smoke_test(device="cpu"):
    torch.manual_seed(0)

    cfg = LVLMConfig(
        image_size=32,
        patch_size=8,
        d_vision = 128,
        d_model = 128,
        n_heads=4,
        n_layers=4,
        vocab_size=200,
        max_text_len=16
    )

    model = MiniLVLM(cfg).to(device)

    B = 2
    images = torch.randn(B, 3, cfg.image_size, cfg.image_size, device=device)
    input_ids = torch.randint(0, cfg.vocab_size, (B,10), device=device)

    print("images : ", images.shape)
    print("input_ids: ", input_ids.shape)
    logits = model(images, input_ids)

    
    print("logits: ", logits.shape)


    assert logits.shape == (B, cfg.n_patches+10, cfg.vocab_size)


if __name__=="__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    phase0_smoke_test(device=device)









