"""
Utils for Transferable Tactile Transformer (T3)

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath
from typing import Optional, Tuple
import numpy as np

def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        # Apple Silicon
        device = "mps"
    else:
        device = "cpu"
    return device

def makeMLP(input_dim,
            output_dim,
            hidden_dims,
            dropout_p,
            tanh_end,
            ln):
    layers = [nn.Linear(input_dim, hidden_dims[0]), nn.SiLU()]
    for i in range(1, len(hidden_dims)):
        layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        if dropout_p > 1e-5:
            layers.append(nn.Dropout(dropout_p))
        if ln:
            layers.append(nn.LayerNorm(hidden_dims[i]))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if tanh_end:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

def makeCNN(input_channels,
            filters,
            kernel_size,
            stride,
            padding):
    layers = [
        nn.Conv2d(input_channels, filters[0], kernel_size=7, stride=2, padding=3, bias=False), 
        nn.ReLU(),
        nn.BatchNorm2d(filters[0]),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    for i in range(1, len(filters)):
        layers.append(nn.Conv2d(filters[i-1], filters[i], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(filters[i]))
    
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)

def findFlattenedSize(input_channels, img_size_x, img_size_y, *nns):
    out = torch.zeros((1, input_channels, img_size_x, img_size_y))
    with torch.no_grad():
        for nn in nns:
            out = nn(out)
        flattened_size = out.shape[1]
    return flattened_size

# --------------------------------------------------------
# Position embedding utils https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
#
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class CrossAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
        query from decoder (x), key and value from encoder (y)
        """
        B, N, C = x.shape
        Ny = y.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop,
        )
        x = attn.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttentionBlock(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.norm1 = norm_layer(decoder_dim)
        self.cross_attn = CrossAttention(
            encoder_dim, decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(decoder_dim)
        mlp_hidden_dim = int(decoder_dim * mlp_ratio)
        self.mlp = Mlp(in_features=decoder_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        """
        x: decoder feature; y: encoder feature (after layernorm)
        """
        x = x + self.drop_path(self.cross_attn(self.norm1(x), y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# --------------------------------------------------------
#    MAE utils for visualization
# --------------------------------------------------------
@torch.no_grad()
def mae_patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))
    return x

@torch.no_grad()
def mae_unpatchify(preds, patch_size):
    """
    preds: (N, L, patch_size**2 *3) for original MAE
    return: (N, 3, H, W)
    """
    h = w = int(preds.shape[1]**.5)
    assert h * w == preds.shape[1]

    preds = preds.reshape(shape=(preds.shape[0], h, w, patch_size, patch_size, 3))
    preds = torch.einsum('nhwpqc->nchpwq', preds)
    imgs = preds.reshape(shape=(preds.shape[0], 3, h * patch_size, h * patch_size))
    return imgs

@torch.no_grad()
def cross_mae_unpatchify(preds, imgs, masks, patch_size):
    """
    preds: (N, l, patch_size**2 *3) for cross MAE
    mask: [N, L], 0 is keep, 1 is remove
    imgs: (N, 3, H, W)
    """
    seq_mask = masks.unsqueeze(2) # (N, L) -> (N, L, 1)
    mask_patch = seq_mask.repeat(1, 1, patch_size**2 * 3) # (N, L, 1) -> (N, L, patch_size**2 *3)
    ori_img_patch = mae_patchify(imgs, patch_size)
    ori_img_patch[mask_patch > 0.5] = preds.reshape(-1)
    return mae_unpatchify(ori_img_patch, patch_size)

@torch.no_grad()
def mae_unpatchify_pred_only(preds, imgs, masks, patch_size):
    """
    apply predicted patches to original images according to masks.
    Difference with mae_unpatchify is that instead of using preds for all patches, we only use preds for patches that are removed.

    preds: (N, L, patch_size**2 *3) for original MAE
    return: (N, 3, H, W)
    """
    assert preds.shape[1] == masks.shape[1]
    preds_removed = preds.masked_select(masks.bool().unsqueeze(-1)).reshape(preds.shape[0], -1, preds.shape[-1])
    return cross_mae_unpatchify(preds_removed, imgs, masks, patch_size)

@torch.no_grad()
def mae_apply_patchified_mask(imgs, masks, patch_size):
    """
    imgs: (N, 3, H, W)
    mask: [N, L], 0 is keep, 1 is remove
    return: (N, 3, H, W), with 1.0 for masked patches
    """
    seq_mask = masks.unsqueeze(2) # (N, L) -> (N, L, 1)
    seq_mask = seq_mask.repeat(1, 1, patch_size**2 * 3) # (N, L, 1) -> (N, L, patch_size**2 *3)
    img_mask = mae_unpatchify(seq_mask, patch_size)
    imgs_ret = imgs.clone()
    imgs_ret[img_mask > 0.5] = 1.0
    
    return imgs_ret