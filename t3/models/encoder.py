"""
Encoder for Transferable Tactile Transformer (T3) models

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""
import os
import torch
from torch import nn
import torchvision
import torch
import torch.nn as nn

import timm.models.vision_transformer as timm_vit
from functools import partial
from .nn_utils import makeCNN, findFlattenedSize, get_2d_sincos_pos_embed, get_device

from t3.utils import logging

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

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
            logging(f"Loading encoder from weights from {path}", True, "green")
            self.load_state_dict(torch.load(path, **kwargs))
        else:
            # try to finetune from gs_green if it exists
            gs_green_path = path[:path.rfind('/')] + '/gs_green.pth'
            if os.path.exists(gs_green_path):
                logging(f"Encoder weights not found at {path}. Loading from gs_green", True, "warning")
                self.load_state_dict(torch.load(gs_green_path, **kwargs))
            else: # if gs_green also doesn't exist, use random initialization
                logging(f"Encoder weights not found at {path}. Skipping", True, "warning")

class IdentityEncoder(Encoder):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class ResNetEncoder(Encoder):
    def __init__(self, 
                 output_dim,
                 model='resnet18',
                 pretrained=True):
        super().__init__()
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        self.model = getattr(torchvision.models, model)(weights=weights)
        self.model.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        return self.model(x)

class CNNEncoder(Encoder):
    def __init__(self,
                 output_dim,
                 input_channels,
                 img_size,
                 filters,
                 kernel_size,
                 stride,
                 padding,
                 **kwargs):
        super(CNNEncoder, self).__init__()
        self.model = makeCNN(input_channels, filters, kernel_size, stride, padding)
        self.flattened_size = findFlattenedSize(input_channels, img_size, img_size, self.model)
        self.fc = nn.Linear(self.flattened_size, output_dim)
    
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

class ViTEncoder(timm_vit.VisionTransformer, Encoder):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float,
                 depth: int,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 **kwargs):
        super(ViTEncoder, self).__init__(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            depth=depth,
            norm_layer=norm_layer,
            **kwargs)
        
        self.blocks = nn.ModuleList([
            timm_vit.Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        del self.head # remove the head
        del self.norm # remove the normalization at the end, which will be added in the trunk

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    def load(self, path):
        """
        Positional embedding interpolation from DeiT
        https://github.com/facebookresearch/deit
        """
        if os.path.exists(path):
            logging(f"Loading encoder from weights from {path}. Will apply pos_embed interpolation.", True, "green")
            checkpoint = torch.load(path, map_location='cpu')
        else:
            gs_green_path = path[:path.rfind('/')] + '/gs_green.pth'
            checkpoint = torch.load(gs_green_path, map_location='cpu')
            logging(f"Encoder weights not found at {path}. Loading from gs_green", True, "warning")
        if 'pos_embed' in checkpoint:
            pos_embed_checkpoint = checkpoint['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = self.pos_embed.shape[-2] - num_patches
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
                checkpoint['pos_embed'] = new_pos_embed
        self.load_state_dict(checkpoint)

class MAEViTEncoder(Encoder):
    """ 
    Masked Autoencoder with VisionTransformer backbone
    https://arxiv.org/pdf/2111.06377.pdf
    """
    def __init__(self, mask_ratio, 
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=3, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.mask_ratio = mask_ratio
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = timm_vit.PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            timm_vit.Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x)

        return (x, mask, ids_restore)