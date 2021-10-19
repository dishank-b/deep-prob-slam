## Depedency
* GTSAM Quadrics - https://github.com/best-of-acrv/gtsam-quadrics
* GTSAM - https://github.com/borglab/gtsam - May not be needed to installed explicitly if installed with gtsam quadrics
* numpy

## Instructions

`./toy_experiments` contains the code that's used to validate that calibrated uncertainty estimates gives better SLAM results. 
    - `python figure8_slam.py` for toy experiments in slam with figure 8 loop closure. 

`./quadric_slam` - containes the code for the reimplementation of quadric slam. 
 - Download the rgb images of the dataset from of the sequence `fr3/long_office_household` at https://vision.in.tum.de/data/datasets/rgbd-dataset/download
 - Put the images under `./quadric_slam/data/rgb/`

```
cd quadric_slam
python quadric_slam.py
```