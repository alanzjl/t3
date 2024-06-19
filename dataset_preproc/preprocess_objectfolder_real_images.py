import os
from utils import WdsWriter
from PIL import Image
from natsort import natsorted
import pandas as pd
import argparse

def data_gen(args):
    domain = "objectfolder_real"
    parent_dir = args.path
    output_dir = args.output_folder
    domain_dir = os.path.join(output_dir, domain)

    labels = pd.read_csv(args.metadata_path, index_col=0, header=0, sep=",")
    material_lookup = {x: i for i, x in enumerate(sorted(list(set(labels["Material"]))))}
    
    img_paths, img_names = [], []
    properties = {
        "obj_idx": [], "obj_name": [],
        "material_idx": [], "material": [], 
        "trial_idx": [],
    }

    for obj_idx in natsorted(os.listdir(parent_dir)):
        if not os.path.isdir(os.path.join(parent_dir, obj_idx)):
            continue
        print(f"Processing obj {obj_idx}...", end="")
        obj_name = labels.loc[int(obj_idx)]["Name"]
        material = labels.loc[int(obj_idx)]["Material"]
        material_idx = material_lookup[material]
        for trial_idx in os.listdir(os.path.join(parent_dir, obj_idx, "tactile_data")):
            if not trial_idx.isdigit():
                continue
            trial_p = os.path.join(parent_dir, obj_idx, "tactile_data", trial_idx)
            if not os.path.isdir(trial_p):
                continue
            gelsight_path = os.path.join(trial_p, "0", "gelsight")
            if not os.path.isdir(gelsight_path):
                print(f"ERROR: path {gelsight_path} does not exist. skipping obj {obj_idx} trial {trial_idx}")
            
            for img_fn in natsorted(os.listdir(gelsight_path)):
                if not img_fn.endswith(".png"):
                    continue
                # this dataset's default image format is png. convert to jpg if not already done
                jpg_fn = img_fn[:-4] + ".jpg"
                jpg_path = os.path.join(gelsight_path, jpg_fn)
                if not os.path.exists(jpg_path):
                    img = Image.open(os.path.join(gelsight_path, img_fn))
                    img.save(jpg_path)

                img_paths.append(jpg_path)
                img_names.append(f"{obj_name}_{trial_idx}_{jpg_fn}")
                properties["obj_idx"].append(obj_idx)
                properties["obj_name"].append(obj_name)
                properties["material_idx"].append(material_idx)
                properties["material"].append(material)
                properties["trial_idx"].append(trial_idx)
        print("done. Size so far: ", len(img_paths))
            
    print(f"Writing {len(img_paths)} images in {domain} to webdataset")
    os.makedirs(domain_dir, exist_ok=True)
    wds = WdsWriter(shard_size=args.shard_size)
    wds.add_imgs(img_paths=img_paths, img_names=img_names)
    wds.add_labels(**properties)
    wds.save(domain_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset pre-processor for ObjectFolder-Real dataset')
    parser.add_argument('-O', '--output_folder', type=str,
                    help='Output folder for the pre-processed dataset')
    parser.add_argument('-S', '--shard_size', type=int, default=10000,
                    help='Maximum number of samples in a single WDS shard.')
    parser.add_argument('--path', type=str,
                    help='Path to the dataset')
    parser.add_argument('--metadata_path', type=str, default="objectfolder_real_metadata.csv",
                    help='Path to the objectfolder_real_metadata.csv metadata file')
    args = parser.parse_args()

    data_gen(args)