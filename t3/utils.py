import torch.distributed as dist
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

color_style = {
    1: "{}",
    2: bcolors.OKBLUE + "{}" + bcolors.ENDC,
    3: bcolors.OKGREEN + "{}" + bcolors.ENDC,
    4: bcolors.OKCYAN + "{}" + bcolors.ENDC,
    0: bcolors.WARNING + "{}" + bcolors.ENDC,
    -1: bcolors.FAIL + "{}" + bcolors.ENDC,
    "blue": bcolors.OKBLUE + "{}" + bcolors.ENDC,
    "green": bcolors.OKGREEN + "{}" + bcolors.ENDC,
    "cyan": bcolors.OKCYAN + "{}" + bcolors.ENDC,
    "warning": bcolors.WARNING + "{}" + bcolors.ENDC,
    "red": bcolors.FAIL + "{}" + bcolors.ENDC,
    "bold": bcolors.BOLD + "{}" + bcolors.ENDC,
}

def logging(s, verbose=True, style=1):
    if not verbose:
        return
    print(color_style[style].format(s))

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def load_label_from_csv(path: str):
    """Load a csv file with pandas"""
    return pd.read_csv(path, index_col=0, header=0, sep=",")

def get_entry_or(cfg, key, default):
    if key in cfg:
        return cfg[key]
    return default

def make_dataset_pie_plot(d, title=None, show=False):
    domains = []
    traj_nums = []
    for k, v in d.items():
        domains.append(f"{k} - {v // 1000}K")
        traj_nums.append(v)
    domains = np.array(domains)
    traj_nums = np.array(traj_nums)
    # sort by number of trajectories
    idx = np.argsort(traj_nums)[::-1]
    domains = domains[idx]
    traj_nums = traj_nums[idx]
    # draw the dataset mixture as a pie plot
    fig1, ax1 = plt.subplots(figsize=(28, 10))
    traj_prob = np.array(traj_nums) / np.sum(traj_nums)
    patches, _ = ax1.pie(traj_prob, startangle=90)
    ax1.axis("equal")
    ax1.legend(patches, domains, loc="center left", bbox_to_anchor=(0.7, 0.5), prop={"size": 25})
    if title is not None:
        ax1.set_title(title, fontsize=60)
    if show:
        plt.show()
    fig1.canvas.draw()
    return Image.frombytes("RGB", fig1.canvas.get_width_height(), fig1.canvas.tostring_rgb())