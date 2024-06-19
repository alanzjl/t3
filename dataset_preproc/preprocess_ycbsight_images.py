import os
from utils import WdsWriter
import json
from scipy.spatial.transform import Rotation as R
from autolab_core import RigidTransform
from natsort import natsorted
import argparse

def calc_real_poses_for_obj(path):
    res = {}
    def d7_to_rotmat(d7, from_frame, to_frame):
        assert len(d7) == 7
        return RigidTransform(
            rotation=R.from_quat([float(x) for x in d7[3:]]).as_matrix(),
            translation=[float(x) for x in d7[:3]],
            from_frame=from_frame,
            to_frame=to_frame
        )
    with open(os.path.join(path, "tf.json"), "r") as f:
        tfs = json.load(f)
    # not sure if this is absolutely correct but this is what the json key says
    robot_to_sensor = d7_to_rotmat(tfs["gripper2gelsight"], "robot", "sensor")
    world_to_obj = d7_to_rotmat(tfs["world2object"], "world", "obj")
    frame_cnt = 0
    with open(os.path.join(path, "robot.csv"), "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split(",")
            # frame_idx = int(line[0])
            robot_pose = d7_to_rotmat(line[1:], "robot", "world") # robot pose in world
            sensor_pose = robot_pose.dot(robot_to_sensor.inverse()) # sensor pose in world
            sensor_to_obj = world_to_obj.dot(sensor_pose) # sensor pose in obj
            assert sensor_to_obj.from_frame == "sensor" and sensor_to_obj.to_frame == "obj"
            res[frame_cnt] = [R.from_matrix(sensor_to_obj.rotation).as_quat(), 1000 * sensor_to_obj.translation]
            frame_cnt += 1
    return res

def calc_sim_poses_for_obj(path):
    res = {}
    def purge_list(l):
        return [x for x in l if x != ""]
    with open(os.path.join(path, "pose.txt"), "r") as f:
        for line in f:
            line = line.strip().split(",")
            frame_idx = int(line[0].split(".")[0])
            quat = [float(x) for x in purge_list(line[1].strip()[1:-1].strip().split(" "))[:4]]
            tra_mm = [1000*float(x) for x in purge_list(line[2].strip()[1:-1].strip().split(" "))[:3]]
            res[frame_idx] = [quat, tra_mm]
    return res

def data_gen(args):
    """Unpack every video under parent_dir"""
    for domain in ["ycbsight_real", "ycbsight_sim"]:
        parent_dir = args.__dict__[domain]
        output_dir = args.output_folder
        domain_dir = os.path.join(output_dir, domain)
        is_real = "real" in domain
        
        img_paths, img_names = [], []
        properties = {
            "x_mm": [], "y_mm": [], "z_mm": [],
            "quat_x": [], "quat_y": [], "quat_z": [], "quat_w": [],
            "obj_idx": [], "obj_name": []
        }

        for obj_idx, obj_name in enumerate(os.listdir(parent_dir)):
            if is_real:
                poses = calc_real_poses_for_obj(os.path.join(parent_dir, obj_name))
                sub_dir = os.path.join(parent_dir, obj_name, "gelsight")
                for frame_idx, img_p in enumerate(natsorted(os.listdir(sub_dir))):
                    img_paths.append(os.path.join(sub_dir, img_p))
                    new_img_fn = f"{obj_name}_{img_p}"
                    img_names.append(new_img_fn)
                    # frame_idx = int(img_p.split("_")[2].split(".")[0])
                    try:
                        properties["x_mm"].append(poses[frame_idx][1][0])
                        properties["y_mm"].append(poses[frame_idx][1][1])
                        properties["z_mm"].append(poses[frame_idx][1][2])
                        properties["quat_x"].append(poses[frame_idx][0][0])
                        properties["quat_y"].append(poses[frame_idx][0][1])
                        properties["quat_z"].append(poses[frame_idx][0][2])
                        properties["quat_w"].append(poses[frame_idx][0][3])
                        properties["obj_idx"].append(obj_idx)
                        properties["obj_name"].append(obj_name)
                    except KeyError:
                        print(f"Frame {frame_idx} not found for {obj_name} in {sub_dir}")
                        exit(1)
            else:
                poses = calc_sim_poses_for_obj(os.path.join(parent_dir, obj_name))
                sub_dir = os.path.join(parent_dir, obj_name, "tactile_imgs")
                for img_p in os.listdir(sub_dir):
                    img_paths.append(os.path.join(sub_dir, img_p))
                    new_img_fn = f"{obj_name}_{img_p}"
                    img_names.append(new_img_fn)
                    frame_idx = int(img_p.split(".")[0])
                    properties["x_mm"].append(poses[frame_idx][1][0])
                    properties["y_mm"].append(poses[frame_idx][1][1])
                    properties["z_mm"].append(poses[frame_idx][1][2])
                    properties["quat_x"].append(poses[frame_idx][0][0])
                    properties["quat_y"].append(poses[frame_idx][0][1])
                    properties["quat_z"].append(poses[frame_idx][0][2])
                    properties["quat_w"].append(poses[frame_idx][0][3])
                    properties["obj_idx"].append(obj_idx)
                    properties["obj_name"].append(obj_name)
        print(f"Writing {len(img_paths)} images in {domain} to webdataset")
        os.makedirs(domain_dir, exist_ok=True)
        wds = WdsWriter(shard_size=args.shard_size)
        wds.add_imgs(img_paths=img_paths, img_names=img_names)
        wds.add_labels(**properties)
        wds.save(domain_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset pre-processor for YCBSight dataset')
    parser.add_argument('-O', '--output_folder', type=str,
                    help='Output folder for the pre-processed dataset')
    parser.add_argument('-S', '--shard_size', type=int, default=10000,
                    help='Maximum number of samples in a single WDS shard.')
    parser.add_argument('--ycbsight_real', type=str,
                    help='Path to the dataset for real images')
    parser.add_argument('--ycbsight_sim', type=str,
                    help='Path to the dataset for simulated images')
    args = parser.parse_args()
    data_gen(args)