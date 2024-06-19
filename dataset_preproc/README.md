# Foundation Tactile (FoTa) Dataset

Note: before running any data preprocessing scripts, make sure to check each script's arguments.

To calculate image normalization of each dataset after preprocessing, use the script `calculate_normalization.py`.

#### [VisGel](http://visgel.csail.mit.edu/) dataset -> 3,170,795 (downsampled to 726,740) tactile images for pretraining (GelSight green sensor)

Download both the seen and unseen dataset from [here](https://github.com/YunzhuLi/VisGel).
Prepare this dataset with 
```sh
python preprocess_visgel_images.py 
```

Useful arguments:

`num_processes=[a number]`: specifies how many processes to create to calculate `cv2.laplacian(diff_img).var()` in parallel. 

`seen_path` and `unseen_path` specify the paths to the data folder.

Note that since a large portion of this dataset is flat images, we further downsample it to roughly 25% of its original size based on a threshold on the variance.

#### [TVL](https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset) dataset -> 82,463 images (DIGIT sensor)

Download data from [here](https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset/tree/main)
Prepare this dataset with 
```sh
python preprocess_tvl_images.py 
```

#### [Touch and Go](https://touch-and-go.github.io/) dataset -> 262,082 images (GelSight TAG sensor, 250,169 with contact)

Download data from [here](https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B)
Prepare this dataset with 
```sh
python preprocess_touchandgo_images.py 
```

This dataset is also used for a classification task (22 classes) with the `datasets/touch_and_go_classification` config.

#### [Calandra corl17 - "More than a feeling"](https://sites.google.com/view/the-feeling-of-success) dataset -> 24,118 images (GelSight green sensor)

Download data from [here](https://drive.google.com/drive/folders/1wHEg_RR8YAQjMnt9r5biUwo5z3P6bjR3)
Prepare this dataset with 
```sh
python preprocess_calandra17_images.py
```
Note that this scripts requires the `deepdish` pip package to handle h5 file operations.

This dataset is also used for a classification task (106 classes) with the `datasets/calandra17_classification` config.

#### [yuan18 - cloth](https://arxiv.org/abs/1711.00574) dataset -> 494,655 images (GelSight green sensor)

Download data from - [here](http://data.csail.mit.edu/active_clothing/Data_ICRA18.tar)
Prepare this dataset with 
```sh
python preprocess_yuan18_images.py
```

This dataset is also used for a three classification tasks: textile_type (20 classes), fuzziness (4 classes), and smoothness (5 classes) with the `datasets/yuan18_classification` config.

#### [YCB-Sight](https://github.com/Robo-Touch/YCB-Sight) dataset -> 480 real images and 1800 sim images (GelSight green sensor)

Download data from - [here](https://drive.google.com/drive/folders/17BPST4biGzduVtoCUBswOmkISqNh1srI)
Prepare this dataset with 
```sh
python preprocess_ycbsight_images.py
```

This dataset also contains object labels and poses. However they are not used for additional tasks other than MAE due to smaller size.

Note that since YCB-Sight Real is significant smaller, we only use it for evaluation during MAE.

#### [ObjectFolder-Real](https://objectfolder.stanford.edu) dataset -> 1,417,600 images (GelSight black sensor)

Download data from - [here](https://objectfolder.stanford.edu/objectfolder-real-download).
Prepare this dataset with 
```sh
python preprocess_objectfolder_real_images.py
```