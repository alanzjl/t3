import os
import numpy as np
from utils import WdsWriter
import deepdish as dd
import cv2
import csv
import pandas as pd
from PIL import Image
import argparse

def extract_frames(dir, extracted_contact_dir, extracted_ref_dir, labels, ref_ratio):
    video1_dir = str(dir) + '/' + 'video.mp4'
    video2_dir = str(dir) + '/' + 'gelsight.mp4'

    cap1 = cv2.VideoCapture(video1_dir)
    frame_number1 = int(cap1.get(7))

    cap2 = cv2.VideoCapture(video2_dir)
    frame_number2 = int(cap2.get(7))
    
    frame_number1 = min(frame_number1, frame_number2)

    for i in range(frame_number1):
        # cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
        # _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        fname = os.path.join(os.path.basename(dir), str(i).rjust(10,'0') + '.jpg')
        if fname in labels.index:
            # has contact
            cv2.imwrite(os.path.join(extracted_contact_dir, fname.replace("/", "-")), frame2)
        else:
            # no contact
            if np.random.rand() < ref_ratio:
                cv2.imwrite(os.path.join(extracted_ref_dir, fname.replace("/", "-")), frame2)
        # cv2.imwrite(str(dir) + '/video_frame/' + str(i).rjust(10,'0') + '.jpg', frame1)
        # cv2.imwrite(str(dir) + '/gelsight_frame/' + str(i).rjust(10,'0') + '.jpg', frame2)

    cap1.release()
    cap2.release()

def data_gen_create_webdataset(args) -> None:
    """
    Create data_gen_create_webdataset
    """
    output_dir = args.output_folder
    domain = "calandra17"
    extracted_dir = os.path.join(args.path, "extracted")

    def _create_img(entry, extracted_dir, dataset, idx, field):
        img = Image.fromarray(entry[field])
        k = f"{os.path.splitext(dataset)[0]}_{idx}_{field}.jpg"
        img.save(os.path.join(extracted_dir, k))
        return [k, entry["object_name"].decode("utf-8"), entry["is_gripping"], "during" in field]

    # extract frames
    os.makedirs(extracted_dir, exist_ok=True)
    labels = [["path", "object_name", "is_gripping", "has_contact"]]

    datasets = [f for f in os.listdir(os.path.join(args.path, "dataset")) if f.endswith(".h5")]
    for dataset in datasets:
        print(f"Extracting frames for {dataset}")
        entries = dd.io.load(os.path.join(args.path, "dataset", dataset))
        for idx, entry in enumerate(entries):
            l = _create_img(entry, extracted_dir, dataset, idx, "gelsightA_during")
            labels.append(l)

            l = _create_img(entry, extracted_dir, dataset, idx, "gelsightB_during")
            labels.append(l)

            if np.random.rand() < args.no_contact_ratio:
                l = _create_img(entry, extracted_dir, dataset, idx, "gelsightA_before")
                labels.append(l)

                l = _create_img(entry, extracted_dir, dataset, idx, "gelsightB_before")
                labels.append(l)
    with open(os.path.join(extracted_dir, "labels.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(labels)

    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)

    img_paths, img_names, has_contact, object_name, is_gripping = [], [], [], [], []
    assert os.path.exists(extracted_dir), f"Extracted dir {extracted_dir} does not exist"
    labels = pd.read_csv(os.path.join(extracted_dir, "labels.csv"), header=0, sep=",")
    for i in range(len(labels)):
        img_paths.append(os.path.join(extracted_dir, labels.iloc[i]["path"]))
        img_names.append(labels.iloc[i]["path"])
        has_contact.append(bool(labels.iloc[i]["has_contact"]))
        object_name.append(labels.iloc[i]["object_name"])
        is_gripping.append(bool(labels.iloc[i]["is_gripping"]))
    
    print(f"Total images: {len(img_paths)}")

    wds = WdsWriter(shard_size=args.shard_size)
    wds.add_imgs(img_paths=img_paths, img_names=img_names)
    wds.add_labels(has_contact=has_contact, object_name=object_name, is_gripping=is_gripping)
    wds.save(domain_dir)
    print("Labels saved for ", domain, " Total images: ", len(img_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset pre-processor for Calandra17 dataset')
    parser.add_argument('-O', '--output_folder', type=str,
                    help='Output folder for the pre-processed dataset')
    parser.add_argument('-S', '--shard_size', type=int, default=10000,
                    help='Maximum number of samples in a single WDS shard.')
    parser.add_argument('--path', type=str,
                    help='Path to the dataset')
    parser.add_argument('--no_contact_ratio', type=float, default=0.3,
                help='This dataset also contains no-contact images. This ratio specifies the fraction of no-contact images to include in the dataset.')
    args = parser.parse_args()
    data_gen_create_webdataset(args)
