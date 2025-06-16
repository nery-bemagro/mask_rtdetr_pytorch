# src/zoo/rtdetr/rtdetr_postprocessor.py (or your equivalent path)
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
        self.remap_mscoco_category = remap_mscoco_category # This flag controls the remapping

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}, remap_mscoco_category={self.remap_mscoco_category}'

    def forward(self, outputs, orig_target_sizes):
        # print("[PostProc INFERENCE DEBUG] --------------- NEW INFERENCE CALL ---------------")
        logits, boxes_cxcywh = outputs['pred_logits'], outputs['pred_boxes']
        pred_masks_logits_all = outputs.get('pred_masks', None) # Raw logits from mask_head, use .get for safety

        # print(f"[PostProc INFERENCE DEBUG] outputs['pred_logits'] shape: {logits.shape}")
        # if pred_masks_logits_all is not None:
        #     print(f"[PostProc INFERENCE DEBUG] outputs['pred_masks'] shape: {pred_masks_logits_all.shape}")
        # else:
        #     print(f"[PostProc INFERENCE DEBUG] outputs['pred_masks'] is None")


        img_w_h = orig_target_sizes.unsqueeze(1) # [B, 1, 2] (W,H for each)
        scale_fct = torch.cat([img_w_h, img_w_h], dim=2) # [B, 1, 4] for (W,H,W,H) scaling
        
        # Convert boxes to absolute xyxy format
        boxes_abs_xyxy = torchvision.ops.box_convert(boxes_cxcywh, 'cxcywh', 'xyxy') * scale_fct

        query_indices = None # Initialize

        if self.use_focal_loss:
            scores_all_classes = logits.sigmoid()
            if self.num_classes == 1: # Special case for single-class detection
                # Squeeze the class dimension if it exists and is 1
                if scores_all_classes.ndim == 3 and scores_all_classes.shape[-1] == 1:
                    scores_all_classes_squeezed = scores_all_classes.squeeze(-1) # [B, N_queries]
                else: # Assumes scores_all_classes is already [B, N_queries]
                    scores_all_classes_squeezed = scores_all_classes
                
                if scores_all_classes_squeezed.shape[1] > self.num_top_queries:
                    scores, query_indices_for_topk = torch.topk(scores_all_classes_squeezed, self.num_top_queries, dim=1)
                else:
                    scores = scores_all_classes_squeezed
                    query_indices_for_topk = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)

                labels = torch.zeros_like(scores, dtype=torch.long) # All labels are 0 for single class
                query_indices = query_indices_for_topk

            else: # Multi-class detection
                scores_flat = scores_all_classes.flatten(1) # [B, N_queries * N_classes]
                
                # Ensure we don't request more queries than available
                k = min(self.num_top_queries, scores_flat.shape[1])
                
                scores, index_flat = torch.topk(scores_flat, k, dim=1)
                labels = index_flat % self.num_classes      # Model's internal label (e.g., 0-79)
                query_indices = index_flat // self.num_classes # Index of the query proposal
        
        else: # Softmax path (usually includes a background class)
            # Exclude background class if present (often the last class)
            scores_all_classes = logits.softmax(-1)[:, :, :-1] 
            _scores_max, _labels_max = scores_all_classes.max(-1) # _labels_max are model's internal labels

            if _scores_max.shape[1] > self.num_top_queries:
                scores, query_indices_for_topk = torch.topk(_scores_max, self.num_top_queries, dim=1)
                labels = torch.gather(_labels_max, dim=1, index=query_indices_for_topk)
                query_indices = query_indices_for_topk
            else:
                scores = _scores_max
                labels = _labels_max
                query_indices = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).repeat(scores.shape[0], 1)

        # ---START LABEL REMAPPING---
        if self.remap_mscoco_category and labels.numel() > 0:
            # IMPORTANT: Ensure this import path is correct for your project structure!
            try:
                from src.data.coco import mscoco_label2category
            except ImportError:
                # Try a common alternative path if the above fails
                # You might need to adjust this based on your specific project structure
                # e.g., if rtdetr_postprocessor is in src/zoo/rtdetr/
                # and coco.py is in src/data/
                from ....data.coco import mscoco_label2category

            # print(f"[PostProc DEBUG] Before remapping - unique labels: {torch.unique(labels).tolist()}")
            original_shape = labels.shape
            flat_labels = labels.flatten()
            
            # Create the remapped labels. Handle potential errors if a label is not in the map.
            remapped_flat_labels_list = []
            for l_item in flat_labels:
                model_label_int = int(l_item.item())
                try:
                    # If mscoco_label2category is a list/array (model label is index)
                    if isinstance(mscoco_label2category, (list, tuple)):
                        coco_category_id = mscoco_label2category[model_label_int]
                    # If mscoco_label2category is a dict (model label is key)
                    elif isinstance(mscoco_label2category, dict):
                        coco_category_id = mscoco_label2category[model_label_int]
                    else:
                        # Should not happen if mscoco_label2category is set up correctly
                        print(f"[PostProc WARNING] mscoco_label2category is of unexpected type: {type(mscoco_label2category)}. Skipping remapping for label {model_label_int}.")
                        coco_category_id = model_label_int # Fallback to original label
                    remapped_flat_labels_list.append(coco_category_id)
                except (IndexError, KeyError) as e:
                    print(f"[PostProc WARNING] Label {model_label_int} not found in mscoco_label2category. Error: {e}. Using original label. Check your mscoco_label2category map and model's output range.")
                    remapped_flat_labels_list.append(model_label_int) # Fallback to original label
            
            labels = torch.tensor(
                remapped_flat_labels_list,
                dtype=labels.dtype, # Keep original dtype if possible, though COCO IDs are usually int
                device=labels.device
            ).reshape(original_shape)
            # print(f"[PostProc DEBUG] After remapping - unique labels: {torch.unique(labels).tolist()}")
        # ---END LABEL REMAPPING---

        # Gather the top boxes corresponding to the selected scores/labels
        # query_indices should be [B, num_top_queries]
        # boxes_abs_xyxy should be [B, N_total_queries, 4]
        if query_indices.max() >= boxes_abs_xyxy.shape[1]:
            print(f"[PostProc WARNING] query_indices.max() ({query_indices.max()}) >= boxes_abs_xyxy.shape[1] ({boxes_abs_xyxy.shape[1]}). This might indicate an issue. Clamping indices.")
            query_indices = torch.clamp(query_indices, max=boxes_abs_xyxy.shape[1] - 1)

        top_boxes = boxes_abs_xyxy.gather(dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, 4))
        
        top_masks_logits = None
        if pred_masks_logits_all is not None and query_indices is not None:
            # pred_masks_logits_all should be [B, N_total_queries, H_mask, W_mask]
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
                "labels": labels[i], # These are now potentially COCO category IDs
                "boxes": top_boxes[i], # Absolute xyxy coordinates
            }

            if top_masks_logits is not None:
                current_masks_logits = top_masks_logits[i] # [num_top_queries, H_mask, W_mask]
                masks_to_resize = current_masks_logits.unsqueeze(1) # [num_top_queries, 1, H_mask, W_mask]
                
                # orig_target_sizes[i, 0] is W_orig, orig_target_sizes[i, 1] is H_orig
                # F.interpolate size expects (H_out, W_out)
                target_h = int(round(orig_target_sizes[i, 1].item())) # Target Height
                target_w = int(round(orig_target_sizes[i, 0].item())) # Target Width
                
                if target_h > 0 and target_w > 0 and masks_to_resize.numel() > 0 : # Ensure valid target size and non-empty masks
                    # print(f"[PostProc DEBUG] Interpolating masks for batch item {i} to H={target_h}, W={target_w}")
                    resized_masks_logits = F.interpolate(
                        masks_to_resize,
                        size=(target_h, target_w), # Tuple of ints (H_out, W_out)
                        mode='bilinear',
                        align_corners=False # Common practice
                    ).squeeze(1) # [num_top_queries, H_orig, W_orig]
                    
                    # Convert to binary masks (0.0 or 1.0)
                    binary_masks = (resized_masks_logits.sigmoid() > 0.5).float()
                    res["masks"] = binary_masks
                elif masks_to_resize.numel() > 0 : # Masks exist but target size is invalid
                    print(f"[PostProc WARNING] Batch item {i}: Invalid target size for mask interpolation H={target_h}, W={target_w}. Skipping mask processing for this item.")
                    res["masks"] = torch.empty((0, target_h if target_h > 0 else 1, target_w if target_w > 0 else 1), device=masks_to_resize.device) # empty masks
                # If masks_to_resize is empty, res["mask"] will be unset, which is fine if no masks were predicted/selected.

            results.append(res)
            
            # Optional: Add the debug print from previous suggestion here if needed
            # if i == 0 and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            # if i == 0 : # For non-distributed or to print on all ranks (can be noisy)
            #     print(f"\n[PostProc EVAL DEBUG] Image {i} (Original Size WxH: {orig_target_sizes[i].tolist()})")
            #     print(f"  Scores (Top 5): {res['scores'][:5].tolist()}")
            #     print(f"  Labels (Top 5 - after remap): {res['labels'][:5].tolist()}")
            #     print(f"  Boxes (Top 5, xyxy absolute): {res['boxes'][:5].tolist()}")
            #     if "masks" in res and res['masks'].numel() > 0:
            #         print(f"  Masks (Top 1 shape): {res['masks'][0].shape if res['masks'].shape[0] > 0 else 'N/A'}")
            #     else:
            #         print(f"  Masks: Not present or empty")

        return results
        
    def deploy(self, ):
        self.eval()
        # self.deploy_mode = False # This was in old code, but not used in the new logic
        return self

    @property
    def iou_types(self, ):
        # If you are only evaluating bounding boxes:
        # return ('bbox', )
        # If you are evaluating segmentation as well:
        return ('bbox', 'segm')