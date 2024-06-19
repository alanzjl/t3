"""
Reconstruction loss for MAE for Transferable Tactile Transformer (T3) pre-training

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""
import torch
import torch.nn as nn

class MAEReconLoss(nn.Module):
    """
    Reconstruction loss for MAE
    """
    def __init__(self, patch_size, norm_pix_loss=False):
        super().__init__()
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, pred, imgs):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3] (original MAE) or [N, l, p*p*3] (cross MAE)
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        (x, mask, ids_restore) = pred
        target = self.patchify(imgs)
        # select the masked portion
        target = target.masked_select(mask.bool().unsqueeze(-1)).reshape(target.shape[0], -1, target.shape[-1])
        if x.shape[1] != target.shape[1]:
            # in the case of original MAE, need to only select the masked portion
            x = x.masked_select(mask.bool().unsqueeze(-1)).reshape(x.shape[0], -1, x.shape[-1])
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (x - target) ** 2
        loss = loss.mean()
        return loss