"""
Calculate image normalization on each dataset
"""

import os
import numpy as np
from PIL import Image

import webdataset as wds
from natsort import natsorted

root_dir = "data/processed"

def print_array(arr):
    return "[" + ", ".join([f"{x:.5f}" for x in arr]) + "]"

def run_sharded_dataset(root_dir, num_samples=50, exlude=[], only=[]):
    # sample num_samples images from the train folder and take the average
    if only:
        ds_names = [os.path.join(root_dir, _) for _ in only]
    else:
        ds_names = [os.path.join(root_dir, _) for _ in os.listdir(root_dir)]
    for ds in ds_names:
        if ds in exlude:
            continue
        if ds.startswith("panda"):
            continue
        print("-" * 80)
        print(f"Dataset: {ds}")
        tars = natsorted([d for d in os.listdir(os.path.join(ds, "train")) if d.startswith("data-")])
        _start_idx = os.path.splitext(os.path.basename(tars[0]))[0].strip("data-")
        _end_idx = os.path.splitext(os.path.basename(tars[-1]))[0].strip("data-")
        url = os.path.join(ds, "train", f"data-{{{_start_idx}..{_end_idx}}}.tar")

        dataset = (
            wds.WebDataset(url, shardshuffle=True)
                .shuffle(1000)
                .decode("pil")
                .to_tuple("jpg", "json")
        )
        imgs = []
        for i, (img, label) in enumerate(dataset):
            if len(imgs) >= num_samples:
                break
            img = np.array(img, dtype=np.float32) / 255.0
            if len(imgs) > 0 and img.shape != imgs[0].shape:
                continue
            imgs.append(img)
        flat_img = np.hstack(imgs)
        print("IMG mean & norm: ", print_array(np.mean(flat_img, axis=(0, 1))), print_array(np.std(flat_img, axis=(0, 1))))

if __name__ == "__main__":
    run_sharded_dataset(root_dir, only=["cnc/cnc_Mini", "cnc/cnc_DenseTact", "cnc/cnc_Svelte"])