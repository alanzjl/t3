"""
Utilities for task-specific processing
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from autolab_core import RigidTransform
import unittest
from ._calandra17_label_dict import CALANDRA17_LABEL_DICT

def process_touch_and_go_label(label):
    """
    Process the classification label for touch_and_go dataset.
    """
    return int(label["contact_class"]) + 2

def process_yuan18_textile_type_label(label):
    """
    Process the textile_type classification label for yuan18 dataset.
    """
    if "textile_type" not in label:
        return int(label["taxtile_type"]) # typo in the dataset
    return int(label["textile_type"])

def process_yuan18_smoothness_label(label):
    """
    Process the smoothness classification label for yuan18 dataset.
    """
    return int(label["smoothness"])

def process_yuan18_fuzziness_label(label):
    """
    Process the fuzziness classification label for yuan18 dataset.
    """
    return int(label["fuzziness"])

def process_calandra17_obj_label(label):
    """
    Process the object classification label for calandra17 dataset.
    106 classes in total.
    """
    if not label["has_contact"]:
        return CALANDRA17_LABEL_DICT["no_contact"]
    return CALANDRA17_LABEL_DICT[label["object_name"]]

def process_objectfolder_real_label(label):
    """
    Process the material classification label for objectfolder_real dataset.
    """
    return int(label["material_idx"])

def process_cnc_cls_label(label):
    """
    Process the classification label for cnc dataset.
    """
    return int(label["obj_idx"])

####################
# Classification processing #
####################

def count_classification_topk(pred, Y, k=5):
    """
    Count top-k success.
    """
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    topk = np.argsort(pred, axis=1)[:, -k:]
    return np.sum(np.any(topk == Y[:, None], axis=1))


####################
# Pose processing #
####################

def rotmat_to_d6(r):
    """
    convert rotation matrix to d6 rotation
    "On the Continuity of Rotation Representations in Neural Networks" https://arxiv.org/abs/1812.07035
    """
    return np.concatenate((r[:,0], r[:, 1]))

def quat_to_d6(q):
    """
    convert quaternion [qx, qy, qz, qw] to d6 rotation
    """
    r = R.from_quat(q).as_matrix()
    return rotmat_to_d6(r)

def pose_to_d9(pose: np.ndarray):
    """
    Convert pose to d9 representation.
    """
    if len(pose.shape) == 1:
        return np.concatenate((pose[:3], quat_to_d6(pose[3:])))
    else:
        res = np.zeros((pose.shape[0], 9))
        res[:, :3] = pose[:, :3]
        for i, q in enumerate(pose[:, 3:]):
            res[i, 3:] = quat_to_d6(q)
        return res

def d6_to_rotmat(d6):
    '''
    Convert 6d representation to rotation matrix.
    '''
    a1, a2 = d6[:3], d6[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=1)

def d6_to_anga(d6):
    '''
    Convert 6d representation to angle-axis representation.
    '''
    rotmat = d6_to_rotmat(d6)
    return R.from_matrix(rotmat).as_rotvec()

def calc_delta_between_pose(in1: np.ndarray, in2: np.ndarray):
    """
    Calculate the delta between two [x, y, z, q_x, q_y, q_z, q_w]. Return a d9 pose.
    Input and output can be batched.
    """
    def _single(a, b):
        mat_1 = np.eye(4)
        mat_1[:3, 3] = a[:3]
        mat_1[:3, :3] = R.from_quat(a[3:]).as_matrix()

        mat_2 = np.eye(4)
        mat_2[:3, 3] = b[:3]
        mat_2[:3, :3] = R.from_quat(b[3:]).as_matrix()
        
        delta = np.linalg.pinv(mat_1).dot(mat_2)
        delta_xyz = delta[:3, 3]
        delta_rot = rotmat_to_d6(delta[:3, :3])
        return np.concatenate((delta_xyz, delta_rot))
    
    if len(in1.shape) == 1:
        return _single(in1, in2)
    else:
        res = np.zeros((in1.shape[0], 9))
        for i, (a, b) in enumerate(zip(in1, in2)):
            res[i] = _single(a, b)
        return res

def d9_normalize(d9: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Normalize d9 pose.
    if mean and std are 3d, only apply normalization to translations.
    """
    res = d9.copy()
    if len(mean) == 3:
        res[:, :3] -= mean
        res[:, :3] /= std
    elif len(mean) == 9:
        res -= mean
        res /= std
    else:
        raise ValueError("Invalid mean and std shape")
    return res

def d9_denormalize(d9: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """
    Denormalize d9 pose.
    if mean and std are 3d, only apply normalization to translations.
    """
    res = d9.copy()
    if len(mean) == 3:
        res[:, :3] *= std
        res[:, :3] += mean
    elif len(mean) == 9:
        res *= std
        res += mean
    else:
        raise ValueError("Invalid mean and std shape")
    return res

@torch.no_grad()
def tra_rmse(pred, gt, denormalize_func=None):
    """
    Calculate the RMSE of translations.
    Both inputs should either be np.ndarray or torch.Tensor.
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if denormalize_func is not None:
        gt = denormalize_func(gt)
        pred = denormalize_func(pred)
    return np.sqrt(np.mean(np.sum((gt[:, :3] - pred[:, :3]) ** 2, axis=1)))

@torch.no_grad()
def rot_rmse(pred, gt, denormalize_func=None):
    """
    Calculate the RMSE of rotations.
    Both inputs should either be np.ndarray or torch.Tensor.
    Assume no normalization done on rotation.
    Output is in degrees.
    TODO: implement 3d rotation RMSE.
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if denormalize_func is not None:
        gt = denormalize_func(gt)
        pred = denormalize_func(pred)
    square_errors = []
    for i in range(gt.shape[0]):
        gt_rot = d6_to_rotmat(gt[i, 3:])
        pred_rot = d6_to_rotmat(pred[i, 3:])
        delta = np.linalg.pinv(gt_rot).dot(pred_rot)
        delta_rotvec = R.from_matrix(delta).as_rotvec(degrees=True)
        square_errors.append(np.sum(delta_rotvec ** 2))
    return np.sqrt(np.mean(square_errors))

def sample_transformation(r_tra, r_rot, from_frame, to_frame):
    '''
    Sample a transformation matrix with range_translation and range_rotation.
    '''
    T = RigidTransform(from_frame=from_frame, to_frame=to_frame)
    T.translation = (np.random.random(3) - 0.5) * 2 * r_tra
    rot = R.from_euler('zyx', (np.random.random(3) - 0.5) * 2 * r_rot, degrees=True)
    T.rotation = rot.as_matrix()
    return T

class TestPoseConversions(unittest.TestCase):
    def test_d6(self):
        rot = sample_transformation(0.1, 10, "world", "camera").rotation
        d6 = rotmat_to_d6(rot)
        self.assertTrue(np.allclose(d6_to_rotmat(d6), rot))

if __name__ == "__main__":
    unittest.main()