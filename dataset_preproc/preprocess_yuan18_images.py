import os
from utils import WdsWriter
import json
import argparse

def data_gen(args):
    parent_dir = args.path
    property_keys = [
        "fuzziness", "thickness", "smoothness", "wool", "stretchiness", "endurance", "softness", 
        "wind_resistance", "season", "wash_method", "textile_type"]
    
    with open(os.path.join(parent_dir, "..", "cloth_metadata.json"), "r") as f:
        material_metadata = json.load(f)

    # extract .mp4 to frames
    for cloth_idx in os.listdir(parent_dir):
        cloth_dir = os.path.join(parent_dir, cloth_idx)
        for trial_idx in os.listdir(cloth_dir):
            trial = os.path.join(cloth_dir, trial_idx)
            vid_p = os.path.join(trial, "GelSight_video.mp4")
            vid_frames_p = os.path.join(trial, "gsframes")
            if os.path.exists(vid_p):
                print(f"Unpacking {vid_p}")
                os.makedirs(vid_frames_p, exist_ok=True)
                os.system(f"ffmpeg -i {vid_p} -vf fps=30 {vid_frames_p}/frame%06d.jpg")
    
    img_paths, img_names = [], []
    properties = {k: [] for k in property_keys}

    for cloth_idx in os.listdir(parent_dir):
        cloth_dir = os.path.join(parent_dir, cloth_idx)
        for trial_idx in os.listdir(cloth_dir):
            trial = os.path.join(cloth_dir, trial_idx)
            vid_frames_p = os.path.join(trial, "gsframes")
            for frame in os.listdir(vid_frames_p):
                if not frame.endswith(".jpg"):
                    continue
                new_frame_name = f"{cloth_idx}_{trial_idx}_{frame}"
                img_paths.append(os.path.join(vid_frames_p, frame))
                img_names.append(new_frame_name)
                for i, k in enumerate(property_keys):
                    properties[k].append(material_metadata[str(cloth_idx)][i])

    print(f"Writing {len(img_paths)} images to webdataset")
    output_dir = args.output_folder
    domain_dir = os.path.join(output_dir, "yuan18")
    os.makedirs(domain_dir, exist_ok=True)
    wds = WdsWriter(shard_size=args.shard_size)
    wds.add_imgs(img_paths=img_paths, img_names=img_names)
    wds.add_labels(**properties)
    wds.save(domain_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset pre-processor for Yuan18 dataset')
    parser.add_argument('-O', '--output_folder', type=str,
                    help='Output folder for the pre-processed dataset')
    parser.add_argument('-S', '--shard_size', type=int, default=10000,
                    help='Maximum number of samples in a single WDS shard.')
    parser.add_argument('--path', type=str,
                    help='Path to the dataset')
    args = parser.parse_args()

    data_gen(args)