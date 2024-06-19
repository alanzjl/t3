import os
import numpy as np
from utils import WdsWriter
import pandas as pd
import cv2
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
    domain = "touch_and_go"

    # load labels
    labels = pd.read_csv(os.path.join(args.path, "label.txt"), header=None, sep=",")
    labels.set_index(0, inplace=True)

    # load label references
    labels_ref = {}
    with open(os.path.join(args.path, "category_reference.txt"), "r") as f:
        for line in f:
            v, k = line.strip().split(":")
            if "(" in k:
                k = k[:k.find("(")].strip()
            labels_ref[int(k)] = v.lower().replace("'", "")

    extracted_contact_dir = os.path.join(args.path, "extracted", "contact")
    extracted_ref_dir = os.path.join(args.path, "extracted", "nocontact")
    
    # extract frames from video
    os.makedirs(extracted_contact_dir, exist_ok=True)
    os.makedirs(extracted_ref_dir, exist_ok=True)

    d = os.path.join(args.path, "dataset")
    folders = [f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
    for folder in folders:
        print(f"Extracting frames from {folder}...")
        extract_frames(os.path.join(d, folder), extracted_contact_dir, extracted_ref_dir, labels, ref_ratio=0.01)


    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)

    img_paths, img_names, has_contact, contact_class, contact_name = [], [], [], [], []
    assert os.path.exists(extracted_contact_dir), f"Extracted contact dir {extracted_contact_dir} does not exist"
    assert os.path.exists(extracted_ref_dir), f"Extracted ref dir {extracted_ref_dir} does not exist"
    for f in os.listdir(extracted_contact_dir):
        img_paths.append(os.path.join(extracted_contact_dir, f))
        img_names.append(f)
        has_contact.append(True)
        class_id = int(labels.loc[f.replace("-", "/")][1])
        if class_id in labels_ref:
            class_name = labels_ref[class_id]
        else:
            class_name = "unknown"
        contact_class.append(class_id)
        contact_name.append(class_name)
    
    N_contact = len(img_paths)
    
    ref_files = [f for f in os.listdir(extracted_ref_dir) if f.endswith(".jpg")]
    np.random.shuffle(ref_files)
    ref_files = ref_files[:int(len(img_paths) * args.no_contact_ratio)]
    for ref_f in ref_files:
        img_paths.append(os.path.join(extracted_ref_dir, ref_f))
        img_names.append(ref_f)
        has_contact.append(False)
        contact_class.append(-2)
        contact_name.append("nocontact")
    
    print(f"Total images: {len(img_paths)} in which {N_contact} has contact")

    wds = WdsWriter(shard_size=args.shard_size)
    wds.add_imgs(img_paths=img_paths, img_names=img_names)
    wds.add_labels(has_contact=has_contact, contact_class=contact_class, contact_name=contact_name)
    wds.save(domain_dir)
    print("Labels saved for ", domain, " Total images: ", len(img_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset pre-processor for Touch-and-Go dataset')
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
