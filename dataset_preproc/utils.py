import os
import webdataset as wds
import json
import numpy as np

class WdsWriter:
    def __init__(self, shard_size, random_order=True):
        self.shard_size = shard_size
        self.random_order = random_order
        self.N = 0

    def add_imgs(self, img_paths, img_names=None):
        """
        Pass in img_names in case a different key is needed for the images. 
        Otherwise, the file name will be used as the key.
        """
        self.N = len(img_paths)
        if self.random_order:
            self.order = np.random.permutation(self.N)
        else:
            self.order = np.arange(self.N)
        self.img_paths = [img_paths[i] for i in self.order]
        if img_names is None:
            self.img_names = [img_path.split('/')[-1] for img_path in self.img_paths]
        else:
            assert len(img_names) == self.N, "number of img_names should be the same as number of images"
            self.img_names = [img_names[i] for i in self.order]
    
    def add_labels(self, **kwargs):
        assert self.N > 0, "add images first"
        for v in kwargs.values():
            assert len(v) == self.N, "number of labels should be the same as number of images"
        self.labels = []
        for idx in self.order:
            self.labels.append({k: kwargs[k][idx] for k in kwargs})
    
    def samples_generator(self, st, ed):
        for i in range(st, ed):
            img_path = self.img_paths[i]
            img_name = self.img_names[i]
            with open(img_path, 'rb') as f:
                img = f.read()
            # key of a sample should not contain '.' or extension
            key = os.path.splitext(img_name)[0].replace('.', '-')
            sample = {
                "__key__": key,
                "jpg": img,
                "json": json.dumps(self.labels[i])
            }
            yield sample
    
    def save(self, output_dir, val_ratio=0.2):
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
        N = len(self.img_paths)
        val_st = int(N * (1 - val_ratio))
        with wds.ShardWriter(os.path.join(output_dir, "train", "data-%06d.tar"), maxcount=self.shard_size) as sink:
            for sample in self.samples_generator(0, val_st):
                sink.write(sample)
        with open(os.path.join(output_dir, "train", "count.txt"), 'w') as f:
            f.write(str(val_st))
        with wds.ShardWriter(os.path.join(output_dir, "val", "data-%06d.tar"), maxcount=self.shard_size) as sink:
            for sample in self.samples_generator(val_st, N):
                sink.write(sample)
        with open(os.path.join(output_dir, "val", "count.txt"), 'w') as f:
            f.write(str(N - val_st))