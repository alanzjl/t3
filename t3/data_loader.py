"""
Data loader for MAE for Transferable Tactile Transformer (T3) using the FoTa dataset

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import hydra
from functools import partial
import os
import numpy as np
from torchvision import transforms
from natsort import natsorted
import webdataset as wds
from .task_utils import calc_delta_between_pose, d9_normalize, d9_denormalize

class TactileImageDatasetBase(IterableDataset):
    def __init__(self, 
                 data_dir: str, 
                 label_func,
                 batch_size: int,
                 encoder_domain: str,
                 decoder_domain: str,
                 img_norm=None,
                 random_resize_crop=True,
                 random_hv_flip_prob=0.5,  # probability of flipping the image
                 color_jitter=None,  # dict with keys brightness, contrast, saturation, hue
                 **kwargs):
        self.encoder_domain = encoder_domain
        self.decoder_domain = decoder_domain
        self.data_dir = data_dir
        self.batch_size = batch_size
        # datasets under data_dir should have the format of data-xxxxxx.tar
        tars = natsorted([d for d in os.listdir(data_dir) if d.startswith("data-")])
        _start_idx = os.path.splitext(os.path.basename(tars[0]))[0].strip("data-")
        _end_idx = os.path.splitext(os.path.basename(tars[-1]))[0].strip("data-")
        with open(os.path.join(data_dir, "count.txt"), "r") as f:
            self.length = int(f.readline())

        url = os.path.join(data_dir, f"data-{{{_start_idx}..{_end_idx}}}.tar")

        img_mean = img_norm["mean"] if img_norm is not None else [0.485, 0.456, 0.406]
        img_std = img_norm["std"] if img_norm is not None else [0.229, 0.224, 0.225]


        preproc = []
        if random_resize_crop:
            preproc.append(transforms.RandomResizedCrop(224))
        else:
            preproc.append(transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True))
            preproc.append(transforms.CenterCrop(224))
        if random_hv_flip_prob > 1e-4:
            preproc.append(transforms.RandomHorizontalFlip(p=random_hv_flip_prob))
            preproc.append(transforms.RandomVerticalFlip(p=random_hv_flip_prob))
        if color_jitter is not None:
            preproc.append(transforms.ColorJitter(**color_jitter))
        
        normalize = transforms.Normalize(
            mean=img_mean,
            std=img_std)
        
        preproc.extend([
            transforms.ToTensor(),
            normalize,
        ])

        # create an inv-normalize function for visualization
        self.inv_normalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(img_mean, img_std)],
            std=[1/s for s in img_std]
        )

        self.dataset = (
            wds.WebDataset(url, shardshuffle=True)
                .shuffle(10000)
                .decode("pil")
                .to_tuple("jpg", "json")
                .map_tuple(transforms.Compose(preproc), label_func)
                .batched(batch_size)
        )
    
    def __len__(self):
        return self.length // self.batch_size
    
    def __iter__(self):
        for img_batch, label_batch in self.dataset:
            # discard the last incomplete batch
            if len(img_batch) < self.batch_size:
                break
            yield {
                "X": img_batch, "Y": label_batch, 
                "encoder_domain": self.encoder_domain, "decoder_domain": self.decoder_domain, 
                "inv_normalize": self.inv_normalize}
    
    def get_dataloader(self, num_workers, **kwargs):
        return DataLoader(self, batch_size=None, shuffle=False, num_workers=num_workers, **kwargs)

class SingleTowerMAEDataset(TactileImageDatasetBase):
    '''
    This dataloader sets both X and Y to be the images. 
    Use this for MAE where the target (Y) is also an image.
    '''
    def __init__(self, **kwargs):
        label_func = lambda x: 0 # a dummy label for mae
        super().__init__(label_func=label_func, **kwargs)
    
    def __iter__(self):
        for img_batch, label_batch in self.dataset:
            # discard the last incomplete batch
            if len(img_batch) < self.batch_size:
                break
            yield {
                "X": img_batch, "Y": img_batch, 
                "encoder_domain": self.encoder_domain, "decoder_domain": self.decoder_domain, 
                "inv_normalize": self.inv_normalize}

class SingleTowerClassificationDataset(TactileImageDatasetBase):
    '''
    This dataloader sets Y based on a label_process_func which extracts a desired label. 
    '''
    def __init__(self, 
                 label_process_func, # a function to process the label
                 **kwargs):
        if isinstance(label_process_func, str):
            label_process_func = hydra.utils.get_method(label_process_func)
        super().__init__(label_func=label_process_func, **kwargs)

class SingleTowerVarianceDataset(TactileImageDatasetBase):
    '''
    This dataloader sets Y to be the variance of laplacian of the images.
    '''
    def __init__(self, **kwargs):
        label_func = lambda x: x["vars"]
        super().__init__(label_func=label_func, **kwargs)
    
    def __iter__(self):
        for img_batch, label_batch in self.dataset:
            # discard the last incomplete batch
            if len(img_batch) < self.batch_size:
                break
            yield {
                "X": img_batch, "Y": torch.from_numpy(label_batch.reshape(-1, 1)).type(torch.FloatTensor), 
                "encoder_domain": self.encoder_domain, "decoder_domain": self.decoder_domain, 
                "inv_normalize": self.inv_normalize}

class DoubleTowerPoseEstimationDataset(TactileImageDatasetBase):
    '''
    This dataloader sets X to be two images and Y to be the difference in pose between the two images.
    The two images in X are randomly rolled to create the pair.
    '''
    def __init__(self, 
                 pose_dim, 
                 label_norm,
                 **kwargs):
        assert pose_dim in [3, 6]
        if pose_dim == 3:
            label_func = lambda x: np.array([x["x_mm"], x["y_mm"], x["z_mm"]])
        else:
            label_func = lambda x: np.array([
                x["x_mm"], x["y_mm"], x["z_mm"],
                x["quat_x"], x["quat_y"], x["quat_z"], x["quat_w"]])
        super().__init__(label_func=label_func, **kwargs)
        self.pose_dim = pose_dim
        self.Y_mean = label_norm["mean"]
        self.Y_std = label_norm["std"]
        # create an inv-normalize function for labels
        if pose_dim == 3:
            self.label_inv_normalize = lambda x: x * self.Y_std + self.Y_mean
        else:
            self.label_inv_normalize = partial(d9_denormalize, mean=self.Y_mean, std=self.Y_std)

    def __iter__(self):
        # For double tower, roll each batch by a random value to get the other images
        for img_batch, label_batch in self.dataset:
            # discard the last incomplete batch
            if len(img_batch) < self.batch_size:
                break
            amount_to_roll = np.random.randint(len(img_batch))
            other_img_batch = torch.roll(img_batch, amount_to_roll, dims=0) # torch.flip(img_batch, dims=[0])
            other_label_batch = np.roll(label_batch, amount_to_roll, axis=0) # np.flip(label_batch, axis=0)

            if self.pose_dim == 3:
                Y = other_label_batch - label_batch
                Y = (Y - self.Y_mean) / self.Y_std
            elif self.pose_dim == 6:
                Y = calc_delta_between_pose(label_batch, other_label_batch)
                Y = d9_normalize(Y, self.Y_mean, self.Y_std)

            Y = torch.from_numpy(Y).type(torch.FloatTensor)
            yield {
                "X": [img_batch, other_img_batch], "Y": Y, 
                "encoder_domain": self.encoder_domain, "decoder_domain": self.decoder_domain, 
                "inv_normalize": self.inv_normalize,
                "label_inv_normalize": self.label_inv_normalize}

class WeightedDataLoader:
    def __init__(self, dataloaders, weight_type="root"):
        """
        This dataloader combines multiple dataloaders into one to be used in training.
        :param dataloaders: list of pytorch dataloaders
        :param weight_type: type of weighting, e.g., "equal", "invlinear", "root"
        """
        self.dataloaders = dataloaders
        datasizes = np.array([len(d) for d in dataloaders], dtype=float)
        if weight_type == 'equal':
            self.weights = np.ones(len(dataloaders)) / len(dataloaders)
        elif weight_type == "invlinear":
            self.weights = (1. / datasizes) / np.sum(1. / datasizes)
        elif weight_type == "root":
            inv_root = np.power(datasizes, 1.0 / 2)
            self.weights = inv_root / np.sum(inv_root)
        else:
            print(f"weight type '{weight_type}' not defined. Using equal weights.")
            self.weights = np.ones(len(dataloaders)) / len(dataloaders)

        self.loader_iters = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        # Choose a dataloader based on weights
        chosen_dataloader_idx = np.random.choice(len(self.dataloaders), p=self.weights)
        chosen_loader_iter = self.loader_iters[chosen_dataloader_idx]
        try:
            return next(chosen_loader_iter)
        except StopIteration:
            # Handle case where a dataloader is exhausted. Reinitialize the iterator.
            self.loader_iters[chosen_dataloader_idx] = iter(self.dataloaders[chosen_dataloader_idx])
            return self.__next__()

    def __len__(self):
        return np.sum([len(dataloader) for dataloader in self.dataloaders])