import pandas as pd
import numpy as np
import os
from tqdm import  tqdm
import csv

def keyframe_det(det_res_root, vid_file):
    f_path = os.path.join(det_res_root, vid_file)
    vid_info = np.load(f_path)

    boxes = vid_info['boxes']
    scores = vid_info['scores']
    masks = vid_info['masks']

    x_c = (boxes[:, 0] + boxes[:, 2]) / 2
    y_c = (boxes[:, 1] + boxes[:, 3]) / 2

    euc_dist = center_dist(x_c, y_c)
    keyframe = frame_detection(scores, euc_dist)

    # return keyframe, boxes[keyframe], masks[keyframe], scores[keyframe]
    return keyframe

def center_dist(xc, yc, C_X=640, C_Y=360):
    return np.sqrt((xc-C_X)**2+(yc-C_Y)**2)


def find_peaks(arr):
    peaks = []
    n = len(arr)

    # Check if the array has fewer than 3 elements; cannot have a peak
    if n < 3:
        return peaks

    # Check the first element
    if arr[0] > arr[1]:
        peaks.append(0)

    # Check for peaks in the middle of the array
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append(i)

    # Check the last element
    if arr[-1] > arr[-2]:
        peaks.append(len(arr)-1)

    return np.array(peaks)


def frame_detection(scores, euc_dist, kf_range=(200, 230)):
    peak_candidates = find_peaks(scores)
    if kf_range:
        l_bound, u_bound = kf_range
        mask = (peak_candidates >= l_bound) & (peak_candidates <= u_bound)
        peak_candidates = peak_candidates[mask]
    if len(peak_candidates) == 0:
        return -1
    min_idx = np.argmin(euc_dist[peak_candidates])
    detected_frame = peak_candidates[min_idx]

    return detected_frame

if __name__=='__main__':
    det_result_root = './det_result'
    vid_list = sorted(os.listdir(det_result_root))
    print(len(vid_list))

    keyframe_list = []
    proc_bar = tqdm(enumerate(vid_list))
    for i, vid_f_name in proc_bar:
        kf_idx = keyframe_det(det_result_root, vid_f_name)
        keyframe_list.append(kf_idx)
        proc_bar.set_description('vid [{}], keyframe[{}]'.format(vid_f_name, kf_idx))

    # Specify the CSV file name
    csv_file_name = 'files_and_values.csv'

    # Writing to the CSV file
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Optionally write the header row, if desired
        writer.writerow(['vid_name', 'keyframe'])

        # Write the data
        for file_name, value in zip(vid_list, keyframe_list):
            writer.writerow([file_name.split('.')[0], value])

    print("CSV file '{csv_file_name}' has been written successfully.")