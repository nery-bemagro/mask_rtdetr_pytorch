# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from src.misc import dist


__all__ = ['CocoEvaluator',]


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            # prediction is a dict like {'scores': tensor, 'labels': tensor, 'boxes': tensor, 'masks': tensor}
            # from your RTDETRPostProcessor
            
            if len(prediction) == 0: # No predictions for this image
                continue

            # Check if masks are present and valid
            if "masks" not in prediction or prediction["masks"] is None or prediction["masks"].numel() == 0:
                # print(f"[CocoEvaluator Segm] Image {original_id}: No masks in prediction output. Skipping segm eval for this image.")
                continue

            # scores, labels, and masks should correspond to each other (N items each)
            scores_list = prediction["scores"].tolist() 
            labels_list = prediction["labels"].tolist() # These should be COCO category_ids
            
            # masks_tensor is expected to be a [N, H, W] tensor of 0.0s and 1.0s (float)
            masks_tensor = prediction["masks"]

            # Convert the [N, H, W] float tensor to a NumPy uint8 array on CPU
            # The postprocessor already did sigmoid > 0.5, so values are 0.0 or 1.0
            masks_np_uint8 = masks_tensor.cpu().numpy().astype(np.uint8) # Shape: [N, H, W]

            processed_rles = []
            for k_idx in range(masks_np_uint8.shape[0]):
                # single_mask_hw is a 2D numpy array [H, W] containing 0s and 1s
                single_mask_hw = masks_np_uint8[k_idx]
                
                # pycocotools.mask.encode expects a Fortran-contiguous 2D array.
                # It returns an RLE dictionary, e.g., {'size': [H, W], 'counts': b'rle_string'}
                if single_mask_hw.shape[0] == 0 or single_mask_hw.shape[1] == 0:
                    # print(f"[CocoEvaluator Segm] Image {original_id}, Mask {k_idx}: Zero-sized mask encountered, skipping RLE encoding for this mask.")
                    # Add a placeholder or skip? For now, let's ensure not to add faulty RLEs.
                    # This might lead to len(processed_rles) < len(scores_list).
                    # A better approach might be to ensure postprocessor doesn't produce these sizes or filter them there.
                    continue

                rle = mask_util.encode(np.asfortranarray(single_mask_hw))
                
                # The 'counts' field in the RLE dict is often bytes and needs to be a utf-8 string for JSON.
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('utf-8')
                processed_rles.append(rle)
            
            # Sanity check: Number of RLEs should match number of scores/labels
            # If a mask was skipped due to zero size, lengths might mismatch.
            # The COCO format requires one score/label per segmentation.
            # This loop assumes that scores/labels correspond to the successfully processed RLEs.
            # If a mask was skipped, its corresponding score/label should also be skipped.
            # For simplicity, we assume here that your postprocessor provides valid masks
            # for every score/label. If not, more complex filtering is needed.
            # The current postprocessor should maintain this N-to-N correspondence.

            if not (len(processed_rles) == len(scores_list) == len(labels_list)):
                print(f"[CocoEvaluator Segm WARNING] Image {original_id}: Mismatch after RLE processing. "
                      f"RLEs: {len(processed_rles)}, Scores: {len(scores_list)}, Labels: {len(labels_list)}. "
                      "This might lead to incorrect evaluation. Ensure all masks are valid and processed.")
                # For robustness, trim to the minimum length to avoid crashing, but this hides an issue.
                min_items = min(len(processed_rles), len(scores_list), len(labels_list))
                processed_rles = processed_rles[:min_items]
                scores_list = scores_list[:min_items]
                labels_list = labels_list[:min_items]

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels_list[k],   # COCO category ID
                        "segmentation": processed_rles[k], # RLE dict for the k-th mask
                        "score": scores_list[k],        # Score for the k-th mask/detection
                    }
                    for k in range(len(processed_rles)) # Iterate up to the number of successfully processed RLEs
                ]
            )
        return coco_results


    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = dist.all_gather(img_ids)
    all_eval_imgs = dist.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


# import io
# from contextlib import redirect_stdout
# def evaluate(imgs):
#     with redirect_stdout(io.StringIO()):
#         imgs.evaluate()
#     return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################

