import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.core import register

@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False  # Initialize deploy mode flag

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}, remap_mscoco_category={self.remap_mscoco_category}, deploy_mode={self.deploy_mode}'

    def forward(self, outputs, orig_target_sizes):
        logits, boxes_cxcywh = outputs['pred_logits'], outputs['pred_boxes']
        pred_masks_logits_all = outputs.get('pred_masks', None)

        # Convert boxes to absolute coordinates
        img_w_h = orig_target_sizes.unsqueeze(1)  # [B, 1, 2]
        scale_fct = torch.cat([img_w_h, img_w_h], dim=2)  # [B, 1, 4]
        boxes_abs_xyxy = torchvision.ops.box_convert(boxes_cxcywh, 'cxcywh', 'xyxy') * scale_fct

        # Process scores and labels
        if self.use_focal_loss:
            scores_all_classes = logits.sigmoid()
            if self.num_classes == 1:
                scores_all_classes_squeezed = scores_all_classes.squeeze(-1) if scores_all_classes.ndim == 3 else scores_all_classes
                if scores_all_classes_squeezed.shape[1] > self.num_top_queries:
                    scores, query_indices_for_topk = torch.topk(scores_all_classes_squeezed, self.num_top_queries, dim=1)
                else:
                    scores = scores_all_classes_squeezed
                    query_indices_for_topk = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)
                labels = torch.zeros_like(scores, dtype=torch.long)
                query_indices = query_indices_for_topk
            else:
                scores_flat = scores_all_classes.flatten(1)
                k = min(self.num_top_queries, scores_flat.shape[1])
                scores, index_flat = torch.topk(scores_flat, k, dim=1)
                labels = index_flat % self.num_classes
                query_indices = index_flat // self.num_classes
        else:
            scores_all_classes = logits.softmax(-1)[:, :, :-1]
            _scores_max, _labels_max = scores_all_classes.max(-1)
            if _scores_max.shape[1] > self.num_top_queries:
                scores, query_indices_for_topk = torch.topk(_scores_max, self.num_top_queries, dim=1)
                labels = torch.gather(_labels_max, dim=1, index=query_indices_for_topk)
                query_indices = query_indices_for_topk
            else:
                scores = _scores_max
                labels = _labels_max
                query_indices = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)

        # Label remapping
        if self.remap_mscoco_category and labels.numel() > 0:
            try:
                from src.data.coco import mscoco_label2category
            except ImportError:
                from ....data.coco import mscoco_label2category
            
            original_shape = labels.shape
            flat_labels = labels.flatten()
            remapped_flat_labels_list = []
            for l_item in flat_labels:
                model_label_int = l_item.item()
                try:
                    if isinstance(mscoco_label2category, (list, tuple)):
                        coco_category_id = mscoco_label2category[model_label_int]
                    elif isinstance(mscoco_label2category, dict):
                        coco_category_id = mscoco_label2category[model_label_int]
                    else:
                        coco_category_id = model_label_int
                    remapped_flat_labels_list.append(coco_category_id)
                except (IndexError, KeyError):
                    remapped_flat_labels_list.append(model_label_int)
            labels = torch.tensor(remapped_flat_labels_list, dtype=labels.dtype, device=labels.device).reshape(original_shape)

        # Safely gather top boxes
        query_indices = torch.clamp(query_indices, max=boxes_abs_xyxy.shape[1] - 1)
        top_boxes = boxes_abs_xyxy.gather(dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, 4))
        
        # Deploy mode: return batch tensors directly
        if self.deploy_mode:
            return scores, labels, top_boxes
        
        # Non-deploy mode: process masks and return per-image results
        results = []
        top_masks_logits = None
        
        # Process masks only in non-deploy mode
        if pred_masks_logits_all is not None and query_indices is not None:
            if query_indices.max() < pred_masks_logits_all.shape[1]:
                index_for_mask_gather = query_indices.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, pred_masks_logits_all.size(-2), pred_masks_logits_all.size(-1))
                top_masks_logits = pred_masks_logits_all.gather(dim=1, index=index_for_mask_gather)
        
        for i in range(logits.shape[0]):
            res = {
                "scores": scores[i],
                "labels": labels[i],
                "boxes": top_boxes[i],
            }
            
            # Process masks if available
            if top_masks_logits is not None:
                current_masks_logits = top_masks_logits[i]
                masks_to_resize = current_masks_logits.unsqueeze(1)
                target_h = int(round(orig_target_sizes[i, 1].item()))
                target_w = int(round(orig_target_sizes[i, 0].item()))
                
                if target_h > 0 and target_w > 0:
                    resized_masks_logits = F.interpolate(
                        masks_to_resize,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    binary_masks = (resized_masks_logits.sigmoid() > 0.5).float()
                    res["masks"] = binary_masks
                else:
                    res["masks"] = torch.empty((0, target_h, target_w), device=masks_to_resize.device)
            
            results.append(res)
        
        return results
        
    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self

    @property
    def iou_types(self):
        return ('bbox', 'segm')