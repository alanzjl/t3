"""
Decoders for Transferable Tactile Transformer (T3) models

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""
import os
import torch
from torch import nn
from typing import Literal

import torch
import torch.nn as nn
from .nn_utils import makeMLP, get_2d_sincos_pos_embed, CrossAttentionBlock, get_device
import timm.models.vision_transformer as timm_vit
from torchvision.models.resnet import BasicBlock

from t3.utils import logging
from math import sqrt

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        kwargs = {}
        if not torch.cuda.is_available():
            kwargs['map_location'] = get_device()
        if os.path.exists(path):
            logging(f"Loading decoder from weights from {path}", True, "green")
            self.load_state_dict(torch.load(path, **kwargs))
        else:
            logging(f"Decoder weights not found at {path}. Skipping", True, "warning")

class IdentityDecoder(Decoder):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class MLPDecoder(Decoder):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_dims,
                 dropout_p=0.1,
                 tanh_end=False,
                 ln=False,
                 transformer_upstream=False, # if True, the input is assumed to be a sequence of tokens
                 pooling_type: Literal['global', 'cls'] = 'cls', # pooling type for transformer upstream
                 **kwargs):
        super().__init__()
        self.transformer_upstream = transformer_upstream
        self.pooling_type = pooling_type

        self.model = makeMLP(input_dim, output_dim, hidden_dims, dropout_p, tanh_end, ln)
        
    def forward(self, x):
        if self.transformer_upstream:
            if self.pooling_type == 'global':
                x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            elif self.pooling_type == 'cls':
                x = x[:, 0]
        return self.model(x)

class PoolingDecoder(Decoder):
    """Only pooling the transformer output"""
    def __init__(self, 
                 pooling_type: Literal['global', 'cls'] = 'cls', # pooling type for transformer upstream
                 **kwargs):
        super().__init__()
        self.pooling_type = pooling_type
        
    def forward(self, x):
        if self.pooling_type == 'global':
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        elif self.pooling_type == 'cls':
            x = x[:, 0]
        return x

class MLPTwoTowerDecoder(Decoder):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_dims,
                 dropout_p=0.1,
                 tanh_end=False,
                 ln=False,
                 **kwargs):
        super().__init__()

        self.model = makeMLP(input_dim, output_dim, hidden_dims, dropout_p, tanh_end, ln)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)

class CNNFCDecoder(Decoder):
    def __init__(self, 
                 inplanes, # input channels of each tower
                 fc_hidden_dims,
                 output_dim,
                 stride,
                 dropout_p=0.1,
                 tanh_end=False,
                 n_tower=2, # number of towers
                 transformer_upstream=False, # if True, the input is assumed to be a sequence of tokens
                 **kwargs):
        super().__init__()
        self.transformer_upstream = transformer_upstream
        self.n_tower = n_tower

        self.norm_layer = nn.BatchNorm2d

        downsample = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=stride, bias=False),
            self.norm_layer(inplanes),
        )

        self.conv_layers = nn.Sequential(
            BasicBlock(
                inplanes, inplanes, stride=stride, downsample=downsample, norm_layer=self.norm_layer
            ),
            BasicBlock(
                inplanes, inplanes, norm_layer=self.norm_layer
            )
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        branch_fc_layers = [nn.Flatten(), nn.Linear(inplanes, inplanes // 2), nn.SiLU()]
        if dropout_p > 0:
            branch_fc_layers.append(nn.Dropout(dropout_p))
        self.branch_fc = nn.Sequential(*branch_fc_layers)

        output_inplanes = (inplanes // 2) * n_tower
        self.output_fc = makeMLP(output_inplanes, output_dim, fc_hidden_dims, dropout_p, tanh_end, ln=False)
    
    def reshape_transformer_input(self, x):
        """ 
        Reshape transformer input (B, T, C) to be a normal image-kind (B, C, H, W) for Conv layers 
        """
        x = x[:, 1:, :] # remove the cls token
        B, T, C = x.shape
        hw = int(sqrt(T))
        assert hw * hw == T, "Input sequence length must be a perfect square"
        x = x.permute(0, 2, 1).reshape(B, C, hw, hw)
        return x

    def branch(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.branch_fc(x)
        return x
        
    def forward(self, *xs):
        assert len(xs) == self.n_tower, f"Expected {self.n_tower} inputs, got {len(xs)}"
        if self.transformer_upstream:
            xs = [self.reshape_transformer_input(x) for x in xs]

        xs = [self.branch(x) for x in xs]
        x = torch.cat(xs, dim=1)
        x = self.output_fc(x)
        return x

class MAEViTDecoder(Decoder):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.num_patches = (img_size // patch_size) ** 2

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            timm_vit.Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # embed tokens
        (x, mask, ids_restore) = x
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return (x, mask, ids_restore)

class CrossMAEViTDecoder(MAEViTDecoder):
    """ 
    CrossMAE with VisionTransformer backbone
    https://arxiv.org/pdf/2401.14391.pdf
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(
            img_size, patch_size, in_chans, embed_dim, decoder_embed_dim, decoder_depth, 
            decoder_num_heads, mlp_ratio, norm_layer, **kwargs)

        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.initialize_weights()
    
    def forward(self, x):
        # embed tokens
        (y, mask, ids_restore) = x

        N, L = ids_restore.shape

        # construct mask tokens 
        x = self.decoder_pos_embed[:, 1:].masked_select(mask.bool().unsqueeze(-1)).reshape(N, -1, self.mask_token.shape[-1])
        x = x + self.mask_token

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, y)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x) # N, L, patch_size**2 *3

        return (x, mask, ids_restore)