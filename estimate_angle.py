# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

# PCA method
import cv2
import numpy as np
import pandas as pd
from math import atan2, degrees
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from tqdm import tqdm

from video_utils import loadVid

# Build the detector and download our pretrained weights
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
predictor = DefaultPredictor(cfg)

BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)

def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

vocabulary = 'custom'
metadata = MetadataCatalog.get("__unused")
# metadata.thing_classes = ['headphone', 'webcam', 'paper', 'coffee'] # Change here to try your own vocabularies!
metadata.thing_classes = ['baseball_bat']
classifier = get_clip_embeddings(metadata.thing_classes)
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)
# Reset visualization threshold
output_score_threshold = 0.5
for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
    predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
    

root = '/content/drive/My Drive/CMU_Pirates_MSCV_Capstone/videos_09-23'
csv_file = '/content/drive/My Drive/Capstone/video_data_0923_bench.csv'
df = pd.read_csv(csv_file)
print(len(df))
# df = df[:3]
est_angle_list = []
confidence_list = []
box_confidence_list = []
for i in tqdm(range(len(df))):
  filtered_row = df.iloc[i]
  if df.iloc[i]['keyframe'] == -1:
    est_angle_list.append(-1)
    confidence_list.append(0)
    box_confidence_list.append(0)
    continue
  vid_name = filtered_row['PitchID']
  # print(filtered_row)
  vid = loadVid(os.path.join(root,vid_name+'.mp4'))
  keyframe = int(filtered_row['keyframe'])
  frame = vid[keyframe, :, :, :]
  # cv2_imshow(frame)
  angle_2d = filtered_row['Angle2d']

  outputs = predictor(frame)
  v = Visualizer(frame[:, :, ::-1], metadata)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  output_img = out.get_image()[:, :, ::-1]
  # cv2_imshow(output_img)

  if len(outputs["instances"].pred_boxes)<1:
    est_angle_list.append(-1)
    confidence_list.append(0)
    box_confidence_list.append(0)
    continue

  score = outputs['instances'].scores[0].cpu().numpy()
  box = outputs["instances"].pred_boxes.tensor[0].cpu().numpy()
  mask = outputs["instances"].pred_masks[0].cpu().numpy()

  # Apply morphological operations to clean the mask
  kernel = np.ones((2,2), np.uint8)
  cleaned_mask = cv2.dilate(mask.astype(np.uint8),kernel,iterations = 1)
  cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

  y_coords, x_coords = np.where(mask)  # Get the indices of all non-zero points
  points = np.column_stack((x_coords, y_coords))  # Stack them as x, y pairs

  # Perform PCA
  pca = PCA(n_components=2)
  pca.fit(points)
  center = pca.mean_
  first_component = pca.components_[0]

  # Calculate orientation angle using the first principal component
  angle = atan2(first_component[1], first_component[0])
  y_coords, x_coords = np.where(mask)  # Get all x, y coordinates where mask == 1
  m = np.tan(angle)
  b = center[1] - m * center[0]  # y = mx + b => b = y - mx
  predicted_y = m * x_coords + b
  r_squared = r2_score(y_coords, predicted_y)

  orientation_angle = degrees(angle)
  orientation_angle = (orientation_angle + 360) % 180
  # print(orientation_angle,degrees(angle),angle,m,r_squared)

  # Visualization and plotting
  new_mask = np.where(cleaned_mask == 0, 1, 0.9)
  line_length = 100  # Increased line length for better visualization

  start_x = int(center[0] - line_length * np.cos(angle))
  start_y = int(center[1] - line_length * np.sin(angle))
  end_x = int(center[0] + line_length * np.cos(angle))
  end_y = int(center[1] + line_length * np.sin(angle))

  est_angle = orientation_angle
  if est_angle>90:
    est_angle = 180-est_angle
  est_angle_list.append(est_angle)
  confidence_list.append(r_squared)
  box_confidence_list.append(score)

save_path = '/content/drive/My Drive/Capstone/video_data_est_0923_v1.csv'
df['est_angle']=est_angle_list
df['angle_confidence']=confidence_list
df['box_confidence']=box_confidence_list
df.to_csv(save_path)