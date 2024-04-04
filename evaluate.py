import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import cv2

from detectron2.config import get_cfg

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo

from video_utils import loadVid

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

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

def saveVid(predictor, vidName, vidSrcRoot, vidResRoot, detResRoot, args, proc_bar = None):
    vid_masks = []
    vid_boxes = []
    vid_scores = []
    video = loadVid(os.path.join(vidSrcRoot, vidName+'.mp4'))
    video = video[args.left_win:args.right_win, :, :]
    if not os.path.exists(os.path.join(detResRoot, vidName)):
        os.mkdir(os.path.join(detResRoot, vidName))

    for i in range(len(video)):
        if proc_bar:
            proc_bar.set_description('[{:03d}/{:03d}] in {}'.format(i+1, len(video), vidName))
        input_frame = video[i, :, :, :]

        predictions, visualized_output = predictor.run_on_image(input_frame)
        pred_scores, pred_boxes, pred_masks = process_predictions(predictions)

        vid_scores.append(pred_scores)
        vid_boxes.append(pred_boxes)
        vid_masks.append(pred_masks.astype(np.uint8))
        # print(i, pred_scores, pred_scores.shape)

    vid_scores = np.stack(vid_scores)
    vid_boxes = np.stack(vid_boxes)
    vid_masks = np.stack(vid_masks)

    save_path = os.path.join(detResRoot, '{}.npz'.format(vidName))
    np.savez(save_path, boxes=vid_boxes, scores=vid_scores, masks=vid_masks)

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
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
    
    # self-defined arguments
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

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    predictor = VisualizationDemo(cfg, args)
    det_result_root = args.det_result_root
    vid_result_root = args.vid_result_root
    vid_source_root = args.vid_source_root

    os.makedirs(det_result_root, exist_ok=True)
    os.makedirs(vid_result_root, exist_ok=True)

    vid_list = sorted(os.listdir(vid_source_root))
    proc_bar = tqdm(range(len(vid_list)))
    for vid_idx in proc_bar:
        vid_name = vid_list[vid_idx].split('.')[0]
        saveVid(predictor, vid_name, vid_source_root, vid_result_root, det_result_root, args, proc_bar=proc_bar)