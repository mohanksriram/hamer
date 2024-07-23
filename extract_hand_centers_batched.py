import os
import json

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from typing import Optional
import pickle

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils import recursive_to

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
from vitpose_model import ViTPoseModel
import json

def tensor_to_array(obj, idx: Optional[int] = None):
    if isinstance(obj, torch.Tensor):
        if idx is not None:
            obj = obj[idx]
        # Convert tensor to numpy array
        return obj.cpu().numpy() if obj.is_cuda else obj.numpy()
    elif isinstance(obj, dict):
        # Recursively handle dictionaries
        return {k: tensor_to_array(v, idx) for k, v in obj.items()}
    else:
        # Return the object as is if it's not a tensor or dictionary
        return obj

class Processor():

    def __init__(self, checkpoint, body_detector, batch_size=1):
        # Download and load checkpoints
        download_models(CACHE_DIR_HAMER)
        model, self.model_cfg = load_hamer(checkpoint)

        # Setup HaMeR model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(self.device)
        model.eval()
        self.model = model

        # Load detector
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
            detector       = DefaultPredictor_Lazy(detectron2_cfg)
        self.detector = detector
        # keypoint detector
        self.cpm = ViTPoseModel(self.device)
        self.batch_size = batch_size

    def process_img_paths(self, img_paths, rescale_factor):
        hand_annotations = {}
        for img_path in img_paths:
            hand_annotations[img_path.parent.name] = {}
        BATCH_SIZE = self.batch_size
        batches = [img_paths[i:i + BATCH_SIZE] for i in range(0, len(img_paths), BATCH_SIZE)]
        for img_batch in batches:
            print(f"running hamer on {len(img_batch)} frames")
            imgs_cv2 = [cv2.imread(str(img_path)) for img_path in img_batch]
            # resize imgs_cv2 to (256, 256)
            imgs_cv2 = [cv2.resize(img, (256, 256)) for img in imgs_cv2]
            # import pdb; pdb.set_trace()
            det_outs = self.detector(imgs_cv2)
            for idx, img_path in enumerate(img_batch):
                det_out = det_outs[idx]
                img_path = img_batch[idx]
                img_cv2 = imgs_cv2[idx]
                # hand_annotations[img_path.parent.name] = {}
                # img_cv2 = cv2.imread(str(img_path))
                # Detect humans in image
                # det_out = self.detector(img_cv2)
                img = img_cv2.copy()[:, :, ::-1]

                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                pred_scores=det_instances.scores[valid_idx].cpu().numpy()

                # Detect human keypoints for each person
                vitposes_out = self.cpm.predict_pose(
                    img,
                    [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
                )
                bboxes = []
                is_right = []
                # import pdb; pdb.set_trace()
                hand_annotations[img_path.parent.name][Path(img_path.name).stem] = { 'left': {}, 'right': {}}

                # Use hands based on hand keypoint detections
                for vitposes in vitposes_out:
                    left_hand_keyp = vitposes['keypoints'][-42:-21]
                    right_hand_keyp = vitposes['keypoints'][-21:]

                    # Rejecting not confident detections
                    keyp = left_hand_keyp
                    valid = keyp[:,2] > 0.5
                    if sum(valid) > 3:
                        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                        bboxes.append(bbox)
                        is_right.append(0)
                    keyp = right_hand_keyp
                    valid = keyp[:,2] > 0.5
                    if sum(valid) > 3:
                        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                        bboxes.append(bbox)
                        is_right.append(1)

                if len(bboxes) == 0:
                    continue

                boxes = np.stack(bboxes)
                # import pdb; pdb.set_trace()
                right = np.stack(is_right)

                # Run reconstruction on all detected hands
                dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

                for batch in dataloader:
                    batch = recursive_to(batch, self.device)
                    with torch.no_grad():
                        out = self.model(batch)
                    # import pdb; pdb.set_trace()
                    multiplier = (2*batch['right']-1)
                    pred_cam = out['pred_cam']
                    pred_cam[:,1] = multiplier*pred_cam[:,1]
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()

                    for i in range(box_center.shape[0]):
                        x = int(box_center[i, 0].item())
                        y = int(box_center[i, 1].item())
                        
                        # check if box is left or right
                        is_right = batch['right'][i].cpu().numpy()
                        if is_right:
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['right']['hand_center'] = (x, y)
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['right']['hand_box_size'] = tensor_to_array(box_size)
                            # import pdb; pdb.set_trace()
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['right']['hamer'] = tensor_to_array(out, i)
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['right']['scaled_focal_length'] = tensor_to_array(scaled_focal_length)
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['right']['img_size'] = tensor_to_array(img_size)
                            # hand_annotations[img_path.parent.name][Path(img_path.name).stem]['right']['hamer'] = tensor_to_array(out, i)
                            
                        else:
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['left']['hand_center'] = (x, y)
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['left']['hand_box_size'] = tensor_to_array(box_size)
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['left']['hamer'] = tensor_to_array(out, i)

                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['left']['scaled_focal_length'] = tensor_to_array(scaled_focal_length)
                            hand_annotations[img_path.parent.name][Path(img_path.name).stem]['left']['img_size'] = tensor_to_array(img_size)
                            # hand_annotations[img_path.parent.name][Path(img_path.name).stem]['right']['hamer'] = tensor_to_array(out, i)
                            

            # save the hand centers to a json file

        # import pdb; pdb.set_trace()
        with open(f'/grogu/user/mohankus/datasets/ego4d_resized_hand_annotations/{img_path.parent.name}.pkl', 'wb') as file:
            pickle.dump(hand_annotations, file)


def main(img_paths, idx):
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--clip_range', type=str, help='Enter a range of numbers as start,stop', default='0,12')

    args = parser.parse_args()

    start, stop = map(int, args.clip_range.split(','))

    # skip run if idx is not in the range
    if idx < start or idx >= stop:
        return

    processor = Processor(args.checkpoint, args.body_detector, args.batch_size)
    processor.process_img_paths(img_paths, args.rescale_factor)

def generate_grouped_image_paths(data, source_dir):
    image_groups = {}
    for entry in data:
        clip_id = entry[0]
        start_frame_id = entry[1]
        stop_frame_id = entry[2]
        
        # Initialize the list for this clip_id if not already
        if clip_id not in image_groups:
            image_groups[clip_id] = []
        
        # Create and add image paths for each frame from start to stop
        for frame_id in range(start_frame_id, stop_frame_id + 1, 4):
            image_path = Path(f"{source_dir}/{clip_id}/{frame_id:06d}.jpg")
            # if image_path.exists():
            image_groups[clip_id].append(image_path)
    
    return image_groups


if __name__ == "__main__":
    json_path = '/grogu/user/mohankus/datasets/r3m_manifest.json'
    source_dir = '/grogu/datasets/ego4d/v1/frames'

    all_subfolders = os.listdir(source_dir)
    with open(json_path, 'r') as file:
        valid_frames = json.load(file)
    grouped_frames = generate_grouped_image_paths(valid_frames, source_dir)
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    # clip_keys = ['6498b686-829f-4e75-9495-78c1e5b4db46', 'c7e8af35-608f-4d18-b752-2f3140f5fcc4']

    for idx, clip in enumerate(grouped_frames.keys()):
        print(f"running hamer on {clip} with {len(grouped_frames[clip])} frames")
        main(grouped_frames[clip], idx)

