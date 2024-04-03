import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import csv
import argparse


def keyframe_det(det_res_root, vid_file, thresh, kf_range):
    f_path = os.path.join(det_res_root, vid_file)
    vid_info = np.load(f_path)

    boxes = vid_info['boxes']
    scores = vid_info['scores']
    # masks = vid_info['masks']

    keyframe = frame_detection(scores, boxes, thresh, kf_range)

    # return keyframe, boxes[keyframe], masks[keyframe], scores[keyframe]
    return keyframe


def center_dist(xc, yc, C_X=640, C_Y=360):
    return np.sqrt((xc - C_X) ** 2 + (yc - C_Y) ** 2)


def get_bbox_info(bboxes):
    x_c = (bboxes[:, 0] + bboxes[:, 2]) / 2
    y_c = (bboxes[:, 1] + bboxes[:, 3]) / 2

    euc_dist = center_dist(x_c, y_c)
    x_length = bboxes[:, 2] - bboxes[:, 0]
    y_length = bboxes[:, 3] - bboxes[:, 1]

    return euc_dist, x_length, y_length


def find_peaks(arr, l_bound=190, r_bound=249):
    peaks = []
    n = len(arr)

    # Check if the array has fewer than 3 elements; cannot have a peak
    if n < 3:
        return peaks

    # Check the first element
    if arr[0] >= arr[1]:
        peaks.append(l_bound)

    # Check for peaks in the middle of the array
    for i in range(1, n - 1):
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            peaks.append(i + l_bound)

    # Check the last element
    if arr[-1] >= arr[-2]:
        peaks.append(r_bound)

    return np.array(peaks)


def context_proposal(conf_seq, peaks, context_window=1, l_bound=190, r_bound=250):
    candidates = []
    for p in peaks:
        for idx in range(max(p - context_window, l_bound), min(p + context_window + 1, r_bound)):
            if conf_seq[idx-l_bound] > 0.4 and idx not in candidates:
                candidates.append(idx)
    return np.asarray(candidates)


def frame_detection(scores, bboxes, thresh, kf_range=(190, 249)):
    lb, ub = kf_range
    peaks = find_peaks(scores, l_bound=lb, r_bound=ub)
    candidates = context_proposal(scores, peaks)
    cen_dist, hor_len, ver_len = get_bbox_info(bboxes)
    # if kf_range:
    #     l_bound, u_bound = kf_range
    #     mask = (peak_candidates >= l_bound) & (peak_candidates <= u_bound)
    #     peak_candidates = peak_candidates[mask]
    selected_indices = np.asarray([idx for idx in candidates if cen_dist[idx-lb] < thresh])

    if len(selected_indices) == 0:
        return -1

    min_idx = np.argmin(cen_dist[selected_indices-lb])
    detected_frame = selected_indices[min_idx]

    return detected_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_result_root",
        help="the path of the root of detection results",
        default='./detic_results/det_23_result',
        type=str
    )
    parser.add_argument(
        "--kf_result_path",
        help="the path to save the keyframe detection results",
        default='./kf_23_pred.csv',
        type=str
    )
    parser.add_argument(
        "--kf_label_path",
        help="the path to the keyframe labels",
        default=None,
    )
    parser.add_argument(
        "--cen_dist_thresh",
        help="threshold for filtering the keyframe candidates",
        default=120,
        type=float
    )
    parser.add_argument(
        "--left_win",
        help="The starting frame index (inclusive) of the keyframe",
        default=190,
        type=int
    )
    parser.add_argument(
        "--right_win",
        help="The ending frame index (not inclusive) of the keyframe",
        default=250,
        type=int
    )
    args = parser.parse_args()

    # det_result_root = './detic_results/det_23_result'
    det_result_root = args.det_result_root
    vid_list = sorted(os.listdir(det_result_root))
    print(len(vid_list))

    keyframe_list = []
    proc_bar = tqdm(enumerate(vid_list))
    for i, vid_f_name in proc_bar:
        kf_idx = keyframe_det(det_result_root, vid_f_name, args.cen_dist_thresh, (args.left_win, args.right_win))
        keyframe_list.append(kf_idx)
        proc_bar.set_description('vid [{}], keyframe[{}]'.format(vid_f_name, kf_idx))

    # df = pd.read_csv('./vid_23_label.csv')
    if args.kf_label_path is not None:
        df = pd.read_csv(args.kf_label_path)

    # Specify the CSV file name
    # csv_file_name = './result_23/vid_23_pred_130.csv'
    csv_file_name = args.kf_result_path

    # Writing to the CSV file
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Optionally write the header row, if desired
        writer.writerow(['vid_name', 'pred', 'gt', 'detected'])

        # Write the data
        for file_name, value in zip(vid_list, keyframe_list):
            if args.kf_label_path is not None:
                gt = df.loc[df['vid_name'] == file_name.split('.')[0], 'gt'].values[0]
                detected = df.loc[df['vid_name'] == file_name.split('.')[0], 'detected'].values[0]
                if detected == -1:
                    gt = -1
            else:
                gt = -100
                detected = -100
            writer.writerow([file_name.split('.')[0], value, gt, detected])

    print("CSV file '{}' has been written successfully.".format(csv_file_name))
