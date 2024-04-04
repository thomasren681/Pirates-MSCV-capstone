# How to use Detic model to run object orientation

Our code is built on the [Detic model](https://github.com/facebookresearch/Detic?tab=readme-ov-file), and its environment dependencies, which can be installed following the instruction of [Detic installation](https://github.com/facebookresearch/Detic/blob/main/docs/INSTALL.md).

Our implementation for object orientation can be divided into three steps:

1. Detect the baseball bat on each single frame of the video using the Detic model
2. Locate the keyframe using the detection results on baseball bat
3. Estimate the angle from the located segmentation mask

## Baseball bat detection

Our code is able to collect the detection results directly from raw videos (e.g. `.mp4` files), so our first steps is to prepare the video data. To do that, we can simply put all the videos we want to extract in one folder and pass the path to that folder to our code. An example would looks like this:

``````
Detic/
└── videos-pirates
    └── videos_09-23
        ├── 00F01C88-4AF3-4679-A7FC-0F41E060C061.mp4
        ├── 01F8CA5C-16A9-4959-A8D3-E849F7244077.mp4
        ├── 02853FB8-DCBA-4260-A237-ECA3C3C7C040.mp4
        ......
``````

Also, don't forget to create a folder for saving the results, our example would be `mkdir detic_results`.

Then, we can run our code by typing `python bat_detection.py --vid_source_root ./videos-pirates/vid_09-23 --vid_result_root ./detic_results/vid_23_result --det_result_root ./detic_results/det_23_result`.

This would collect the detection results information for further use, and it should look like this (for sanity check):

``````
detic_results/
├── det_23_result
│   ├── 00F01C88-4AF3-4679-A7FC-0F41E060C061.npz
│   ├── 01F8CA5C-16A9-4959-A8D3-E849F7244077.npz
│   ├── 02853FB8-DCBA-4260-A237-ECA3C3C7C040.npz
│   ├── 044FB165-76BD-4656-AC06-40785DD09609.npz
|	......
└── vid_23_result
    ├── 00F01C88-4AF3-4679-A7FC-0F41E060C061
    |	├── 190.jpg
    |	├── 191.jpg
    |	......
    |	├── 249.jpg
    ├── 01F8CA5C-16A9-4959-A8D3-E849F7244077
    ├── 02853FB8-DCBA-4260-A237-ECA3C3C7C040
    ├── 044FB165-76BD-4656-AC06-40785DD09609
``````

## Keyframe detection

The keyframe detection part needs to choose a threshold (only hyperparameter) to perform, and returns a `.csv` file to record the prediction and ground truth label for each video. Our example would be run `python kf_angle_prediction.py --det_result_root ./detic_results/det_23_result --kf_result_path ./kf_23_pred.csv --kf_label_path ./vid_23_label.csv --cen_dist_thresh 120`.

If there is no ground truth label for this dataset, just skip that argument, and the ground truth will be set to `-100` in the output csv file.

Ideally, we will get a csv file recording the keyframe prediction for each video.





