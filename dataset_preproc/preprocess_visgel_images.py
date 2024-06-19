import os
import numpy as np
from multiprocessing import Pool

from natsort import natsorted
from utils import WdsWriter
import webdataset as wds
import cv2
import json
import argparse


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def gen_variance(args):
    path, n_folder, rec_folder = args
    ref_img = "frame0000.jpg"
    if not os.path.isfile(os.path.join(path, n_folder, rec_folder, ref_img)):
        print("Ref image does not exist!")
        print(os.path.join(path, n_folder, rec_folder, ref_img))
        return

    ref_img = cv2.imread(os.path.join(path, n_folder, rec_folder, ref_img))
    gray_float_ref = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY).astype(float)
    weight_dict = {}
    for fn in os.listdir(os.path.join(path, n_folder, rec_folder)):
        if fn.endswith(".jpg"):
            des_img = cv2.imread(os.path.join(path, n_folder, rec_folder, fn))
            gray_float_des = cv2.cvtColor(des_img, cv2.COLOR_RGB2GRAY).astype(float)
            weight = variance_of_laplacian(gray_float_des - gray_float_ref)
            weight_dict[fn] = weight
    np.save(os.path.join(path, n_folder, rec_folder, "variance.npy"), weight_dict)
    print("Variance calculated and saved for ", os.path.join(path, n_folder, rec_folder))

def data_gen_calc_variance_only(args) -> None:
    """
    Calulate variance only without moving images
    """
    arg_lst = []
    for path in [args.seen_path, args.unseen_path]:
        for n_folder in os.listdir(path):
            for rec_folder in os.listdir(os.path.join(path, n_folder)):
                arg_lst.append((path, n_folder, rec_folder))
    
    print("Calculating variance for ", len(arg_lst), " folders")
    with Pool(processes=args.num_processes) as pool:
        print("Using ", pool._processes, " processes")
        pool.map(gen_variance, arg_lst)

def data_gen_create_webdataset(args) -> None:
    """
    Create data_gen_create_webdataset
    """
    output_dir = args.output_folder
    domain = "visgel"
        
    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)

    img_paths, img_names, vars = [], [], []
    for path in [args.seen_path, args.unseen_path]:
        for n_folder in os.listdir(path):
            print(f"Processing {domain} - {path} - {n_folder}")
            for rec_folder in os.listdir(os.path.join(path, n_folder)):
                var_dict = np.load(os.path.join(path, n_folder, rec_folder, "variance.npy"), allow_pickle=True).item()
                for img_fn in var_dict:
                    prefix = "seen" if (path == args.seen_path) else "unseen"
                    new_fn = f"{prefix}_{n_folder}_{rec_folder}_{img_fn}"
                    # os.system(f"ln -s {os.path.join(path, n_folder, rec_folder, img_fn)} {os.path.join(domain_dir, new_fn)}")
                    img_names.append(new_fn)
                    img_paths.append(os.path.join(path, n_folder, rec_folder, img_fn))
                    vars.append(var_dict[img_fn])
    wds = WdsWriter(shard_size=args.shard_size)
    wds.add_imgs(img_paths=img_paths, img_names=img_names)
    wds.add_labels(vars=vars)
    wds.save(domain_dir)
    print("Labels saved for ", domain, " Total images: ", len(img_names))

def downsample_visgel(args) -> None:
    """
    Turns out that VisGel contains quite a lot flat images.
    This script downsamples the images to roughly 1/2 of the original size to make the training more balanced.
    """

    def identity(x):
        return x
    
    output_dir = os.path.join(args.output_folder, "visgel_downsampled")
    
    for sub_dir in ["train", "val"]:
        ori_dir = os.path.join(args.output_folder, "visgel", sub_dir)
        # datasets under data_dir should have the format of data-xxxxxx.tar
        tars = natsorted([d for d in os.listdir(ori_dir) if d.startswith("data-")])
        _start_idx = os.path.splitext(os.path.basename(tars[0]))[0].strip("data-")
        _end_idx = os.path.splitext(os.path.basename(tars[-1]))[0].strip("data-")
        with open(os.path.join(ori_dir, "count.txt"), "r") as f:
            ori_cnt = int(f.readline())
        url = os.path.join(ori_dir, f"data-{{{_start_idx}..{_end_idx}}}.tar")
        ori_dataset = (
            wds.WebDataset(url)
                .decode("pil")
                .to_tuple("__key__", "jpg", "json")
                .map_tuple(identity, identity, identity)
                .shuffle(1000)
        )

        # all_vars = []
        # for i, (key, image, data) in enumerate(ori_dataset):
        #     all_vars.append(data["vars"])
        #     if i > 100000:
        #         break
        # all_vars = np.array(all_vars)
        # print(f"median val: {np.median(all_vars)}, mean val: {np.mean(all_vars)}")
        ## median val: 15.168044748454403, mean val: 19.698067996121345 

        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
        data_cnt = 0
        with wds.ShardWriter(os.path.join(output_dir, sub_dir, "data-%06d.tar"), maxcount=args.shard_size) as sink:
            for key, image, data in ori_dataset:
                sample = {
                    "__key__": key,
                    "jpg": image,
                    "json": json.dumps(data),
                }
                if data["vars"] > 18:
                    data_cnt += 1
                    if data_cnt % 100000 == 0:
                        print(f"...donsampling {data_cnt} images from {ori_cnt} for {sub_dir}")
                    sink.write(sample)
                    continue
        with open(os.path.join(output_dir, sub_dir, "count.txt"), 'w') as f:
            f.write(str(data_cnt))
        print(f"Downsampled {data_cnt} images from {ori_cnt} for {sub_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset pre-processor for VisGel dataset')
    parser.add_argument('-O', '--output_folder', type=str,
                    help='Output folder for the pre-processed dataset')
    parser.add_argument('-S', '--shard_size', type=int, default=10000,
                    help='Maximum number of samples in a single WDS shard.')
    parser.add_argument('-N', '--num_processes', type=int, default=1,
                    help='Number of processes to use for variance calculation.')
    parser.add_argument('--seen_path', type=str,
                    help='Path to the seen part of the dataset')
    parser.add_argument('--unseen_path', type=str,
                    help='Path to the unseen part of the dataset')
    parser.add_argument('--no_downsampling', action='store_true',
                    help='Avoid downsampling the images if specified.')
    args = parser.parse_args()

    data_gen_calc_variance_only(args)
    data_gen_create_webdataset(args)
    if not args.no_downsampling:
        downsample_visgel(args)
