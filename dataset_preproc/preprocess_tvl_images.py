import os
import numpy as np
from utils import WdsWriter
import pandas as pd
import json
import argparse

def data_gen_create_webdataset(args) -> None:
    """
    Create data_gen_create_webdataset
    """
    output_dir = args.output_folder
    domain = "tvl"
    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    img_paths, img_names, has_contact, desc = [], [], [], []

    # Add in the HCT dataset
    for path in [os.path.join(args.hct_path, f"data{i}") for i in range(1, 4)]:
        train_dict = pd.read_csv(os.path.join(path, "train.csv"), header=0, sep=",")
        test_dict = pd.read_csv(os.path.join(path, "test.csv"), header=0, sep=",")
        d = pd.concat([train_dict, test_dict], ignore_index=True)
        for i in range(len(d)):
            img_paths.append(os.path.join(path, d.iloc[i]["tactile"]))
            img_names.append(f"{path.split('/')[-1]}_{d.iloc[i]['tactile'].split('/')[-1]}")
            has_contact.append(True)
            desc.append(d.iloc[i]["caption"])
        # sample non-contact images
        N = int(args.hct_no_contact_ratio * len(d))
        with open(os.path.join(path, "not_contact.json"), "r") as f:
            non_contact = json.load(f)["tactile"]
            for fn in np.random.choice(non_contact, N, replace=False):
                img_paths.append(os.path.join(path, fn))
                img_names.append(f"{path.split('/')[-1]}_{fn.split('/')[-1]}")
                has_contact.append(False)
                desc.append("No contact")
        print("Processing {} for the HCT dataset...".format(path))
    print("Processed HCT dataset. Total images so far: ", len(img_names))
    
    # Add in the SSVTP dataset
    d = pd.read_csv(os.path.join(args.ssvtp_path, "train.csv"), index_col=0, header=0, sep=",")
    for i in range(len(d)):
        img_paths.append(os.path.join(args.ssvtp_path, d.iloc[i]["tactile"]))
        img_names.append(f"ssvtp_{d.iloc[i]['tactile'].split('/')[-1]}")
        has_contact.append(True)
        desc.append(d.iloc[i]["caption"])
    print("Processed SSVTP dataset. Total images so far: ", len(img_names))

    wds = WdsWriter(shard_size=args.shard_size)
    wds.add_imgs(img_paths=img_paths, img_names=img_names)
    wds.add_labels(has_contact=has_contact, desc=desc)
    wds.save(domain_dir)
    print("Labels saved for ", domain, " Total images: ", len(img_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset pre-processor for TVL dataset')
    parser.add_argument('-O', '--output_folder', type=str,
                    help='Output folder for the pre-processed dataset')
    parser.add_argument('-S', '--shard_size', type=int, default=10000,
                    help='Maximum number of samples in a single WDS shard.')
    parser.add_argument('--hct_path', type=str,
                    help='Path to the HCT dataset')
    parser.add_argument('--ssvtp_path', type=str,
                    help='Path to the SSVTP dataset')
    parser.add_argument('--hct_no_contact_ratio', type=float, default=1.0,
                help='The HCT dataset also contains no-contact images. This ratio specifies the fraction of no-contact images to include in the dataset.')
    args = parser.parse_args()
    data_gen_create_webdataset(args)
