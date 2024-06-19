"""
Transferable Tactile Transformer (T3)

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""

import hydra

import os
from torch import nn
from t3.utils import logging

class T3(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.encoders = {}
        self.decoders = {}
        self.loss_funcs = {}
        self.trunk = hydra.utils.instantiate(cfg.shared_trunk)
        self._is_trunk_transformer = "Transformer" in cfg.shared_trunk._target_

        for name, encoder_cfg in cfg.encoders.items():
            self.encoders[name] = hydra.utils.instantiate(encoder_cfg)
        
        for name, decoder_cfg in cfg.decoders.items():
            self.decoders[name] = hydra.utils.instantiate(decoder_cfg)
            if hasattr(decoder_cfg, "loss_func"):
                self.loss_funcs[name] = hydra.utils.instantiate(decoder_cfg.loss_func)
            else:
                self.loss_funcs[name] = None
        
        self.encoders = nn.ModuleDict(self.encoders)
        self.decoders = nn.ModuleDict(self.decoders)
        self.loss_funcs = nn.ModuleDict(self.loss_funcs)
        self._encoder_domain = None
        self._decoder_domain = None
    
    def model_summary(self):
        print("==========================================")
        encoder_parameters = sum(p.numel() for p in self.encoders.parameters() if p.requires_grad)
        trunk_parameters = sum(p.numel() for p in self.trunk.parameters() if p.requires_grad)
        decoder_parameters = sum(p.numel() for p in self.decoders.parameters() if p.requires_grad)
        n_parameters = encoder_parameters + trunk_parameters + decoder_parameters
        logging(
            f"number of total trainable params (M): {n_parameters / 1.0e6:.3f} \n\
                encoder: {encoder_parameters / 1.0e6:.3f} \n\
                    trunk: {trunk_parameters / 1.0e6:.3f} \n\
                        decoder: {decoder_parameters / 1.0e6:.3f}", True, "green")
    
    def set_domains(self, encoder_domain, decoder_domain, forward_mode):
        assert encoder_domain in self.encoders, f"encoder domain {encoder_domain} not found in encoders"
        assert decoder_domain in self.decoders, f"decoder domain {decoder_domain} not found in decoders"
        self._encoder_domain = encoder_domain
        self._decoder_domain = decoder_domain
        self._forward_mode = forward_mode
    
    def freeze_encoder(self, encoder_domain=None):
        if encoder_domain is None:
            for encoder in self.encoders.values():
                encoder.freeze()
        else:
            assert encoder_domain in self.encoders, f"encoder domain {encoder_domain} not found in encoders"
            self.encoders[encoder_domain].freeze()
    
    def unfreeze_encoder(self, encoder_domain=None):
        if encoder_domain is None:
            for encoder in self.encoders.values():
                encoder.unfreeze()
        else:
            assert encoder_domain in self.encoders, f"encoder domain {encoder_domain} not found in encoders"
            self.encoders[encoder_domain].unfreeze()
    
    def freeze_trunk(self):
        self.trunk.freeze()
    
    def unfreeze_trunk(self):
        self.trunk.unfreeze()

    def forward(self, *args, **kwargs):
        if self._forward_mode == "single_tower":
            return self.single_tower_forward(*args, **kwargs)
        elif self._forward_mode == "multi_tower":
            return self.multi_tower_forward(*args, **kwargs)
        else:
            raise ValueError(f"forward mode {self._forward_mode} not recognized")
    
    def single_tower_forward(self, x):
        x = self.encoders[self._encoder_domain](x)
        x = self.trunk(x)
        x = self.decoders[self._decoder_domain](x)
        return x
    
    def multi_tower_forward(self, *xs):
        xs = [self.encoders[self._encoder_domain](x) for x in xs]
        xs = [self.trunk(x) for x in xs]
        x = self.decoders[self._decoder_domain](*xs)
        return x
    
    def compute_loss(self, y_pred, y_true):
        return self.loss_funcs[self._decoder_domain](y_pred, y_true)
    
    def save_components(self, dir):
        os.makedirs(f"{dir}/encoders", exist_ok=True)
        os.makedirs(f"{dir}/decoders", exist_ok=True)
        for encoder_name, encoder in self.encoders.items():
            encoder.save(f"{dir}/encoders/{encoder_name}.pth")
        for decoder_name, decoder in self.decoders.items():
            decoder.save(f"{dir}/decoders/{decoder_name}.pth")
        self.trunk.save(f"{dir}/trunk.pth")

    def load_components(self, dir):
        for encoder_name, encoder in self.encoders.items():
            encoder.load(f"{dir}/encoders/{encoder_name}.pth")
        for decoder_name, decoder in self.decoders.items():
            decoder.load(f"{dir}/decoders/{decoder_name}.pth")
        self.trunk.load(f"{dir}/trunk.pth")

def make_T3_tiny(cfg):
    return T3(cfg)