# Third-party components
# BRAU-Net++: https://github.com/Caipengzhou/BRAU-Netplusplus
# BiFormer: https://github.com/rayleizhu/BiFormer
#
# MIT License
#
# Copyright (c) 2023 Caipengzhou
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from typing import Tuple

import torch
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["DGSCNet"]

class TopkRouting(nn.Module):
    
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv

class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,kv_per_win=4, kv_downsample_ratio=4,
                 kv_downsample_kernel=None, kv_downsample_mode='identity', topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=5,
                 auto_pad=False):
        super().__init__()
        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim

        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5


        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        assert not (self.param_routing and not self.diff_routing)
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:
            mul_weight = 'soft'
        elif self.diff_routing:
            mul_weight = 'hard'
        else:
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'Unsupported kv_downsample_mode: {self.kv_downsample_mode}')

        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad=auto_pad

    def forward(self, x, ret_attn_mask=False):
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,
                          pad_l, pad_r,
                          pad_t, pad_b))
            _, H, W, _ = x.size()
        else:
            N, H, W, C = x.size()
            assert H%self.n_win == 0 and W%self.n_win == 0


        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        q, kv = self.qkv(x)

        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3])

        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)


        r_weight, r_idx = self.router(q_win, k_win)

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)

        out = out + lepe
        out = self.wo(out)

        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out


class BiFormerBlock(nn.Module):

    def __init__(self, dim, resolution, topk, drop_path):
        super().__init__()
        self.input_resolution = resolution
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = BiLevelRoutingAttention(
            dim=dim,
            num_heads=dim // 32,
            n_win=7,
            qk_dim=dim,
            kv_downsample_mode="identity",
            topk=topk,
            param_attention="qkvo",
            param_routing=False,
            diff_routing=False,
            soft_routing=False,
            side_dwconv=5,
            auto_pad=False,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 3 * dim),
            nn.Identity(),
            nn.GELU(),
            nn.Linear(3 * dim, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        height, width = self.input_resolution
        batch, _, channels = x.shape
        x = self.norm1(x).view(batch, height, width, channels).permute(0, 3, 1, 2)
        x = (x + self.pos_embed(x)).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)


class DecoderStage(nn.Module):
    def __init__(self, dim, resolution, depth, topk, upsample):
        super().__init__()
        rates = torch.linspace(0, 0.2, depth).tolist()
        self.blocks = nn.ModuleList([
            BiFormerBlock(dim, resolution, topk, rate) for rate in rates
        ])
        self.upsample = PatchExpand(resolution, dim) if upsample else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.upsample(x) if self.upsample is not None else x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FirstPatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        b, h, w, c = x.shape
        x = x.view(b, h, w, c)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=c // 4)
        x = x.view(b, -1, c // 4)
        return self.norm(x)


class FinalPatchExpandX4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        h, w = self.input_resolution
        x = self.expand(x)
        b, length, c = x.shape
        assert length == h * w, "input feature has wrong size"
        x = x.view(b, h, w, c)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=c // (self.dim_scale ** 2),
        )
        x = x.view(b, -1, self.output_dim)
        return self.norm(x)


class DGSR(nn.Module):

    def __init__(self, dim, window_size=7, bands=3, topk=4, reduction=4):
        super().__init__()
        if bands < 2:
            raise ValueError("bands must be >= 2")
        self.dim = dim
        self.window_size = window_size
        self.bands = bands
        self.topk = topk
        hidden_dim = max(dim // reduction, 16)
        self.q_proj = nn.Linear(dim, hidden_dim)
        self.k_proj = nn.Linear(dim, hidden_dim)
        self.v_proj = nn.Linear(dim, hidden_dim)
        self.band_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, bands))
        self.channel_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim))
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, guide_tokens, target_tokens):
        batch, length, channels = target_tokens.shape
        side = int(math.sqrt(length))
        if side * side != length:
            raise ValueError(f"Token length {length} is not a square feature map")
        if guide_tokens.shape != target_tokens.shape:
            raise ValueError(
                f"guide and target must have same shape, got {guide_tokens.shape} "
                f"and {target_tokens.shape}"
            )
        guide = guide_tokens.transpose(1, 2).reshape(batch, channels, side, side)
        target = target_tokens.transpose(1, 2).reshape(batch, channels, side, side)
        routed = self.forward_nchw(guide, target)
        return routed.flatten(2).transpose(1, 2)

    def forward_nchw(self, guide, target):
        dtype = target.dtype
        batch, channels, height, width = target.shape
        window_size = min(self.window_size, height, width)
        guide_windows, meta = _window_partition(guide.float(), window_size)
        target_windows, _ = _window_partition(target.float(), window_size)
        band_windows = self._split_bands(target_windows, window_size)
        num_windows = meta["num_windows"]
        band_windows = band_windows.view(
            batch, num_windows, self.bands, channels, window_size, window_size
        )
        guide_windows = guide_windows.view(
            batch, num_windows, channels, window_size, window_size
        )

        guide_desc = guide_windows.mean(dim=(-1, -2))
        band_desc = band_windows.abs().mean(dim=(-1, -2))
        query = self.q_proj(guide_desc)
        key = self.k_proj(band_desc).reshape(batch, num_windows * self.bands, -1)
        value = self.v_proj(band_desc).reshape(batch, num_windows * self.bands, -1)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        route_k = min(self.topk, scores.shape[-1])
        top_scores, top_indices = torch.topk(scores, k=route_k, dim=-1)
        route_weight = torch.softmax(top_scores, dim=-1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, value.shape[-1])
        gathered = torch.gather(
            value.unsqueeze(1).expand(-1, num_windows, -1, -1), 2, gather_index
        )
        context = (route_weight.unsqueeze(-1) * gathered).sum(dim=2)

        band_weight = torch.softmax(self.band_head(context), dim=-1).view(
            batch, num_windows, self.bands, 1, 1, 1
        )
        channel_gate = torch.sigmoid(self.channel_head(context)).view(
            batch, num_windows, channels, 1, 1
        )
        residual = (band_weight * band_windows).sum(dim=2) * channel_gate
        residual = residual.reshape(
            batch * num_windows, channels, window_size, window_size
        )
        residual = _window_reverse(residual, meta)
        residual = self.out_proj(residual).to(dtype=dtype)
        return target + self.gamma.to(dtype=dtype) * residual

    def _split_bands(self, windows, window_size):
        frequency = torch.fft.rfft2(windows, dim=(-2, -1), norm="ortho")
        masks = _frequency_band_masks(
            window_size, self.bands, windows.device, windows.dtype
        )
        band_frequency = frequency.unsqueeze(1) * masks.unsqueeze(0)
        return torch.fft.irfft2(
            band_frequency,
            s=(window_size, window_size),
            dim=(-2, -1),
            norm="ortho",
        )


def _window_partition(x, window_size):
    batch, channels, height, width = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    padded_h, padded_w = x.shape[-2:]
    x = x.view(
        batch,
        channels,
        padded_h // window_size,
        window_size,
        padded_w // window_size,
        window_size,
    )
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(-1, channels, window_size, window_size)
    return windows, {
        "batch": batch,
        "channels": channels,
        "height": height,
        "width": width,
        "padded_height": padded_h,
        "padded_width": padded_w,
        "window_size": window_size,
        "num_windows": (padded_h // window_size) * (padded_w // window_size),
    }


def _window_reverse(windows, meta):
    batch = meta["batch"]
    channels = meta["channels"]
    height = meta["height"]
    width = meta["width"]
    padded_h = meta["padded_height"]
    padded_w = meta["padded_width"]
    window_size = meta["window_size"]
    x = windows.view(
        batch,
        padded_h // window_size,
        padded_w // window_size,
        channels,
        window_size,
        window_size,
    )
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(
        batch, channels, padded_h, padded_w
    )
    return x[:, :, :height, :width]


def _frequency_band_masks(window_size, bands, device, dtype):
    frequency_y = torch.fft.fftfreq(window_size, device=device)
    frequency_x = torch.fft.rfftfreq(window_size, device=device)
    radius = torch.sqrt(frequency_y[:, None] ** 2 + frequency_x[None, :] ** 2)
    radius = radius / radius.max().clamp_min(1e-6)
    masks = []
    for band_index in range(bands):
        low = band_index / bands
        high = (band_index + 1) / bands
        if band_index == 0:
            mask = radius <= high
        elif band_index == bands - 1:
            mask = radius > low
        else:
            mask = (radius > low) & (radius <= high)
        masks.append(mask.to(dtype=dtype))
    return torch.stack(masks, dim=0)[:, None, :, :]


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.GELU(),
    )


class CADC(nn.Module):

    def __init__(self, dim, num_classes):
        super().__init__()
        hidden_dim = 96
        token_dim = 48
        self.sdf_tau = 0.25
        self.dose_prior = 0.25

        self.coarse_head = nn.Sequential(
            _conv_block(dim, hidden_dim), nn.Conv2d(hidden_dim, num_classes, 1)
        )
        self.sdf_head = nn.Sequential(
            _conv_block(dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes - 1, 1),
            nn.Tanh(),
        )

        self.spatial_token_proj = nn.Conv2d(dim, token_dim, 1, bias=False)
        self.class_queries = nn.Parameter(torch.empty(1, num_classes, token_dim))
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        self.class_attention = nn.MultiheadAttention(
            token_dim, 4, dropout=0.0, batch_first=True
        )
        self.class_norm1 = nn.LayerNorm(token_dim)
        self.class_ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.class_norm2 = nn.LayerNorm(token_dim)
        self.similarity_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        self.evidence_encoder = _conv_block(dim + 4 * num_classes + 1, hidden_dim)
        self.direction_head = nn.Sequential(
            _conv_block(hidden_dim, hidden_dim), nn.Conv2d(hidden_dim, num_classes, 1)
        )
        dose_hidden = hidden_dim // 2
        self.local_dose_head = nn.Sequential(
            nn.Conv2d(hidden_dim, dose_hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(dose_hidden), dose_hidden),
            nn.GELU(),
            nn.Conv2d(dose_hidden, num_classes, 1),
        )
        self.class_dose_head = nn.Linear(token_dim, 1)

    def reset_identity(self):
        nn.init.zeros_(self.direction_head[-1].weight)
        nn.init.zeros_(self.direction_head[-1].bias)
        nn.init.zeros_(self.local_dose_head[-1].weight)
        nn.init.zeros_(self.local_dose_head[-1].bias)
        nn.init.zeros_(self.class_dose_head.weight)
        prior_logit = math.log(self.dose_prior / (1.0 - self.dose_prior))
        nn.init.constant_(self.class_dose_head.bias, prior_logit)

    def forward(self, feature):
        batch, _, height, width = feature.shape

        coarse_logits = self.coarse_head(feature)
        coarse_probability = torch.softmax(coarse_logits, dim=1)
        top2 = coarse_probability.topk(k=2, dim=1).values
        uncertainty = 1.0 - (top2[:, :1] - top2[:, 1:2])

        foreground_sdf = self.sdf_head(feature)
        background_sdf = -foreground_sdf.amin(dim=1, keepdim=True)
        signed_distance = torch.cat((background_sdf, foreground_sdf), dim=1)
        boundary_gate = torch.exp(-signed_distance.abs() / self.sdf_tau)

        spatial_tokens = self.spatial_token_proj(feature).flatten(2).transpose(1, 2)
        queries = self.class_queries.expand(batch, -1, -1)
        attended, _ = self.class_attention(
            queries, spatial_tokens, spatial_tokens, need_weights=False
        )
        class_tokens = self.class_norm1(queries + attended)
        class_tokens = self.class_norm2(class_tokens + self.class_ffn(class_tokens))
        pixels = F.normalize(spatial_tokens, dim=-1, eps=1e-6)
        classes = F.normalize(class_tokens, dim=-1, eps=1e-6)
        similarity = torch.einsum("bnd,bkd->bkn", pixels, classes)
        similarity_scale = self.similarity_scale.clamp(max=math.log(100.0)).exp()
        similarity = (similarity * similarity_scale).view(batch, -1, height, width)

        evidence = torch.cat(
            (feature, coarse_probability, uncertainty, signed_distance, boundary_gate, similarity),
            dim=1,
        )
        encoded = self.evidence_encoder(evidence)
        raw_direction = self.direction_head(encoded)
        common_mode = (coarse_probability.detach() * raw_direction).sum(1, keepdim=True)
        direction = raw_direction - common_mode

        local_dose_logits = self.local_dose_head(encoded)
        class_dose_logits = self.class_dose_head(class_tokens).squeeze(-1)
        dose = torch.sigmoid(local_dose_logits + class_dose_logits[:, :, None, None])
        correction = boundary_gate * dose * direction

        return {
            "coarse_logits": coarse_logits,
            "foreground_sdf": foreground_sdf,
            "signed_distance": signed_distance,
            "uncertainty": uncertainty,
            "boundary_gate": boundary_gate,
            "class_similarity": similarity,
            "class_tokens": class_tokens,
            "direction": direction,
            "dose": dose,
            "correction": correction,
            "refined_logits": coarse_logits + direction,
        }


class DGSCNetBackbone(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        channels = (96, 192, 384, 768)
        encoder_depths = (2, 2, 8)
        backbone_topk = (2, 4, 8)
        self.patches_resolution = (img_size // 4, img_size // 4)
        resolutions = [(img_size // scale,) * 2 for scale in (4, 8, 16, 32)]

        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.GELU(),
                nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(96),
            )
        ])
        for in_channels, out_channels in zip(channels, channels[1:]):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
            ))

        rates = torch.linspace(0, 0.2, sum(encoder_depths)).tolist()
        self.stages = nn.ModuleList()
        start = 0
        for level, depth in enumerate(encoder_depths):
            self.stages.append(nn.Sequential(*[
                BiFormerBlock(channels[level], resolutions[level], backbone_topk[level], rate)
                for rate in rates[start:start + depth]
            ]))
            start += depth

        self.layers_up = nn.ModuleList([
            FirstPatchExpand(resolutions[3], dim=channels[3])
        ])
        self.concat_back_dim = nn.ModuleList([nn.Identity()])
        for level in (2, 1, 0):
            dim = channels[level]
            self.concat_back_dim.append(nn.Linear(2 * dim, dim))
            self.layers_up.append(DecoderStage(
                dim, resolutions[level], encoder_depths[level], backbone_topk[level],
                upsample=level > 0,
            ))

        self.dgsr_skip_blocks = nn.ModuleDict({
            str(index): DGSR(channels[3 - index], window_size=7, bands=3, topk=4)
            for index in (1, 2, 3)
        })
        self.norm_up = nn.LayerNorm(channels[0])
        self.up4 = FinalPatchExpandX4(self.patches_resolution, dim=channels[0])
        self.output = nn.Conv2d(channels[0], num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

        self.casd_refine = CADC(channels[0], num_classes)
        self.casd_refine.apply(self._init_weights)
        self.casd_refine.reset_identity()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward_features(self, x):
        skips = []
        for level, stage in enumerate(self.stages):
            x = self.downsample_layers[level](x)
            batch, channels, height, width = x.shape
            tokens = stage(x.flatten(2).transpose(1, 2))
            skips.append(tokens)
            x = tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return self.downsample_layers[3](x), skips

    def forward_up_features(self, x, skips):
        x = self.layers_up[0](x)
        for index in (1, 2, 3):
            skip = self.dgsr_skip_blocks[str(index)](x, skips[3 - index])
            x = self.concat_back_dim[index](torch.cat((x, skip), dim=-1))
            x = self.layers_up[index](x)
        return self.norm_up(x)

    def forward(self, x, return_aux=False):
        x, skips = self.forward_features(x)
        tokens = self.forward_up_features(x, skips)
        batch, _, channels = tokens.shape
        height, width = self.patches_resolution

        expanded = self.up4(tokens).view(batch, 4 * height, 4 * width, channels)
        base_logits = self.output(expanded.permute(0, 3, 1, 2))
        feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        aux = self.casd_refine(feature)
        correction = F.interpolate(
            aux["correction"], size=base_logits.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        logits = base_logits + correction
        if return_aux:
            return {**aux, "logits": logits, "base_logits": base_logits, "correction_logits": correction}
        return logits


class DGSCNet(nn.Module):

    def __init__(self, num_classes=9, img_size=224):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground class")
        if img_size < 224 or img_size % 224:
            raise ValueError("img_size must be a positive multiple of 224")
        self.img_size = img_size
        self.bra_unet = DGSCNetBackbone(img_size, num_classes)

    def forward(self, x, return_aux=False):
        if x.ndim != 4 or x.shape[1] not in (1, 3):
            raise ValueError("expected a batch of one-channel or three-channel images")
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"expected square inputs of size {self.img_size}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.bra_unet(x, return_aux=return_aux)
