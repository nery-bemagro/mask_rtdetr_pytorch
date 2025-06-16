# src/core/postprocessor/rtdetr_postprocessor.py (or wherever it's located)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from src.core import register

# __all__ = ['RTDETRPostProcessor'] # Keep this if you use it for imports

@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']

    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category
        # self.deploy_mode = False # Not used here

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes):
        # print("[PostProc INFERENCE DEBUG] --------------- NEW INFERENCE CALL ---------------") # Keep for now
        logits, boxes_cxcywh = outputs['pred_logits'], outputs['pred_boxes']
        pred_masks_logits_all = outputs['pred_masks'] # Raw logits from mask_head

        # print(f"[PostProc INFERENCE DEBUG] outputs['pred_logits'] shape: {logits.shape}")
        # print(f"[PostProc INFERENCE DEBUG] outputs['pred_masks'] shape: {pred_masks_logits_all.shape}")
        
        img_w_h = orig_target_sizes.unsqueeze(1) # [B, 1, 2] (W,H for each)
        scale_fct = torch.cat([img_w_h, img_w_h], dim=2) # [B, 1, 4] for (W,H,W,H) scaling
        boxes_abs_xyxy = torchvision.ops.box_convert(boxes_cxcywh, 'cxcywh', 'xyxy') * scale_fct

        query_indices = None # Initialize
        if self.use_focal_loss:
            scores_all_classes = logits.sigmoid()
            if self.num_classes == 1:
                scores, query_indices = torch.topk(scores_all_classes.squeeze(-1), self.num_top_queries, dim=1)
                labels = torch.zeros_like(scores, dtype=torch.long)
            else:
                scores_flat = scores_all_classes.flatten(1)
                scores, index_flat = torch.topk(scores_flat, self.num_top_queries, dim=1)
                labels = index_flat % self.num_classes
                query_indices = index_flat // self.num_classes
        else: # Softmax path
            scores_all_classes = logits.softmax(-1)[:, :, :-1]
            _scores_max, _labels_max = scores_all_classes.max(-1)
            if _scores_max.shape[1] > self.num_top_queries:
                scores, query_indices = torch.topk(_scores_max, self.num_top_queries, dim=1)
                labels = torch.gather(_labels_max, dim=1, index=query_indices)
            else:
                scores = _scores_max
                labels = _labels_max
                query_indices = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)

        top_boxes = boxes_abs_xyxy.gather(dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, 4))
        
        top_masks_logits = None
        if pred_masks_logits_all is not None and query_indices is not None:
            if query_indices.max() < pred_masks_logits_all.shape[1]:
                index_for_mask_gather = query_indices.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, pred_masks_logits_all.size(-2), pred_masks_logits_all.size(-1)
                )
                top_masks_logits = pred_masks_logits_all.gather(dim=1, index=index_for_mask_gather)
            else:
                print(f"[PostProcessor WARNING] query_indices.max() {query_indices.max()} >= pred_masks_logits_all.shape[1] {pred_masks_logits_all.shape[1]}. Skipping mask gather.")

        results = []
        for i in range(logits.shape[0]): # Iterate over batch size
            res = {
                "scores": scores[i],
                "labels": labels[i],
                "boxes": top_boxes[i],
            }

            if top_masks_logits is not None:
                current_masks_logits = top_masks_logits[i]
                masks_to_resize = current_masks_logits.unsqueeze(1)
                
                # orig_target_sizes[i, 0] is W_orig, orig_target_sizes[i, 1] is H_orig
                # F.interpolate size expects (H_out, W_out)
                target_h = int(round(orig_target_sizes[i, 1].item())) # Height
                target_w = int(round(orig_target_sizes[i, 0].item())) # Width
                
                # print(f"[PostProc DEBUG] Interpolating masks for batch item {i} to H={target_h}, W={target_w}")

                resized_masks_logits = F.interpolate(
                    masks_to_resize,
                    size=(target_h, target_w), # Tuple of ints
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                
                binary_masks = (resized_masks_logits.sigmoid() > 0.5).float()
                res["masks"] = binary_masks
            
            results.append(res)
            
        return results

    def deploy(self, ):
        self.eval()
        return self