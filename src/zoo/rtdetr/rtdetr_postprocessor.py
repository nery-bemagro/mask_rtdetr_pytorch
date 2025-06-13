"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from src.core import register


__all__ = ['RTDETRPostProcessor']


@register
class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        masks = outputs['pred_masks']

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.coco import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)
                
        if masks is not None:
            if self.use_focal_loss:
                # Select top masks using the same indices
                masks = masks.gather(1, index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, masks.size(-2), masks.size(-1)))
            else:
                masks = masks.gather(1, index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, masks.size(-2), masks.size(-1)))

        results = []
        for i, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            res = dict(labels=lab, boxes=box, scores=sco)
            if masks is not None:
                # Resize mask to original image size
                img_mask = masks[i].unsqueeze(1)  # [Q, 1, H_mask, W_mask]
                orig_size = orig_target_sizes[i].tolist()
                resized_mask = F.interpolate(
                    img_mask, 
                    size=orig_size, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)  # [Q, H_orig, W_orig]
                res['masks'] = (resized_mask > 0.5).float()
            results.append(res)
            
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = False
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )
