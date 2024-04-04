import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
from tqdm import tqdm
import sys
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

def get_start_vid(vid_res_root, vid_data_root):
    vid_res_list = sorted(os.listdir(vid_res_root))
    vid_data_list = sorted(os.listdir(vid_data_root))
    # print(vid_data_list)
    i = 0

    for _ in range(len(vid_data_list)):
        if i < len(vid_res_list) and vid_res_list[i] == vid_data_list[i].split('.')[0]:
        # if vid_res_list[i] == vid_data_list[i].split('.')[0]:
            i += 1
            continue
        else:
            break

    s_idx = max(i-1, 0)
    # print(s_idx)
    print('start from {:03d}th video, [{}]'.format(s_idx, vid_res_list[s_idx]))
    return s_idx

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="custom",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default='baseball_bat',
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'],
        nargs=argparse.REMAINDER,
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
    parser.add_argument(
        "--vid_source_root",
        help="The path to the video root",
        type=str
    )
    parser.add_argument(
        "--det_result_root",
        help="The path to the video root",
        default='./detic_results/det_23_result',
        type=str
    )
    parser.add_argument(
        "--vid_result_root",
        help="The path to the video root",
        default='./detic_results/vid_23_result',
        type=str
    )
    return parser

def loadVid(path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    cap = cv2.VideoCapture(path)
    frames = []

    if cap.isOpened() == False:
        return frames
        print("Error opening video [{}]".format(path))
        
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
        
    cap.release()
    frames = np.stack(frames)
    return frames

def writeVideo(video, file_name):
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # writer = cv2.VideoWriter('../result/ar.avi', fourcc, 25.0, [video.shape[2], video.shape[1]], True)
    writer = cv2.VideoWriter(file_name, fourcc, 60.0, [video.shape[2], video.shape[1]], True)

    for f_idx in range(video.shape[0]):
        writer.write(video[f_idx])

    writer.release()

def saveFrame(outputFr, vidName, frameIdx, detResRoot, verbose=False):
    # root = os.path.join('./result', vidName)
    root = os.path.join(detResRoot, vidName)
    fields = outputFr['instances'].get_fields()
    pred_masks = fields['pred_masks'].to('cpu').numpy()
    pred_scores = fields['scores'].to('cpu').numpy()
    pred_boxes = fields['pred_boxes'].tensor.to('cpu').numpy()
    pred_classes = fields['pred_classes'].to('cpu').numpy()

    masks_path = os.path.join(root, '{:04}_masks.npy'.format(frameIdx))
    scores_path = os.path.join(root, '{:04}_scores.npy'.format(frameIdx))
    boxes_path = os.path.join(root, '{:04}_boxes.npy'.format(frameIdx))
    classes_path = os.path.join(root, '{:04}_classes.npy'.format(frameIdx))

    np.save(masks_path, pred_masks)
    np.save(scores_path, pred_scores)
    np.save(boxes_path, pred_boxes)
    np.save(classes_path, pred_classes)

    if verbose:
        print('Video : [{}]\'s {}th frame\'s output has been saved successfully'.format(vidName, frameIdx))

def process_predictions(predictions):
    fields = predictions['instances'].get_fields()
    pred_masks = fields['pred_masks'].to('cpu').numpy()
    pred_scores = fields['scores'].to('cpu').numpy()
    pred_boxes = fields['pred_boxes'].tensor.to('cpu').numpy()
    # pred_classes = fields['pred_classes'].to('cpu').numpy()

    if len(pred_scores) > 1:
        idx = np.argmax(pred_scores)
        pred_masks = pred_masks[idx]
        pred_boxes = pred_boxes[idx]
        pred_scores = pred_scores[idx]
    elif len(pred_scores) == 0:
        pred_scores = np.array([0])
        pred_boxes = np.zeros((4,))
        pred_masks = np.zeros((720, 1280))

    return pred_scores.squeeze(), pred_boxes.squeeze(), pred_masks.squeeze()

def saveVid(demo, vidName, vidSrcRoot, vidResRoot, detResRoot, args, proc_bar = None):
    vid_masks = []
    vid_boxes = []
    vid_scores = []
    video = loadVid(os.path.join(vidSrcRoot, vidName+'.mp4'))
    video = video[args.left_win:args.right_win, :, :]
    if not os.path.exists(os.path.join(vidResRoot, vidName)):
        os.mkdir(os.path.join(vidResRoot, vidName))
    # if not os.path.exists(os.path.join(detResRoot, vidName)):
    #     os.mkdir(os.path.join(detResRoot, vidName))

    for i in range(len(video)):
        if proc_bar:
            proc_bar.set_description('[{:03d}/{:03d}] in {}'.format(i+1, len(video), vidName))
        input_frame = video[i, :, :, :]

        predictions, visualized_output = demo.run_on_image(input_frame)
        # print(type(visualized_output), visualized_output)
        vid_result_path = os.path.join(vidResRoot, vidName, '{:03d}.jpg'.format(i+args.left_win))
        visualized_output.save(vid_result_path)

        pred_scores, pred_boxes, pred_masks = process_predictions(predictions)

        vid_scores.append(pred_scores)
        vid_boxes.append(pred_boxes)
        vid_masks.append(pred_masks.astype(np.uint8))
        # print(i, pred_scores, pred_scores.shape)

    vid_scores = np.stack(vid_scores)
    vid_boxes = np.stack(vid_boxes)
    vid_masks = np.stack(vid_masks)

    # print(scores_arr.shape, boxes_arr.shape, masks_arr.shape)

    save_path = os.path.join(detResRoot, '{}.npz'.format(vidName))
    np.savez(save_path, boxes=vid_boxes, scores=vid_scores, masks=vid_masks)
    # return scores_arr, boxes_arr, masks_arr


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)
    # det_result_root = './det_23_result'
    # vid_result_root = './vid_23_result'
    # vid_source_root = './vid_23/videos_09-23'
    det_result_root = args.det_result_root
    vid_result_root = args.vid_result_root
    vid_source_root = args.vid_source_root

    os.makedirs(det_result_root, exist_ok=True)
    os.makedirs(vid_result_root, exist_ok=True)

    # start_idx = get_start_vid(vid_result_root, vid_source_root)
    # vid_list = sorted(os.listdir(vid_source_root))[start_idx:]
    vid_list = sorted(os.listdir(vid_source_root))
    proc_bar = tqdm(range(len(vid_list)))
    for vid_idx in proc_bar:
        vid_name = vid_list[vid_idx].split('.')[0]
        saveVid(demo, vid_name, vid_source_root, vid_result_root, det_result_root, args, proc_bar=proc_bar)

