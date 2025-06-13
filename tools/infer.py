import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)
  
def draw(images, labels, boxes, scores, masks_list=None, thrh=0.6, path=""):
    """
    Draw bounding boxes, labels, scores, and masks on images.
    Masks are assumed to be already binarized (0 or 1 values).
    """
    # Define colors for different classes (for boxes only)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Create a default font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, im in enumerate(images):
        # Create a copy to draw on
        img = im.copy().convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Get detections above threshold
        scr = scores[i]
        valid_idx = scr > thrh
        lab = labels[i][valid_idx]
        box = boxes[i][valid_idx]
        scrs = scores[i][valid_idx]
        
        # Draw masks first (so boxes appear on top)
        if masks_list is not None and i < len(masks_list) and masks_list[i] is not None:
            masks = masks_list[i][valid_idx]
            
            # Create a black layer for all masks
            black_layer = Image.new("RGBA", img.size, (0, 0, 0, 255))
            mask_combined = Image.new("L", img.size, 0)
            
            # Combine all masks for this image
            for mask in masks:
                # Convert binary mask to PIL Image (0-255)
                mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                # Paste onto combined mask
                mask_combined.paste(255, (0, 0), mask_img)
            
            # Apply combined mask to black layer
            black_layer.putalpha(mask_combined)
            # Composite with original image
            img = Image.alpha_composite(img, black_layer)
            draw = ImageDraw.Draw(img, "RGBA")  # Refresh draw object
        
        # Draw bounding boxes and labels
        for j, b in enumerate(box):
            color = ImageColor.getrgb(colors[int(lab[j]) % len(colors)])
            
            # Draw bounding box
            draw.rectangle(list(b), outline=color + (255,), width=2)
            
            # Draw label
            text = f"{int(lab[j])}: {scrs[j]:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_bg = [
                b[0], b[1],
                b[0] + (text_bbox[2]-text_bbox[0]) + 4,
                b[1] + (text_bbox[3]-text_bbox[1]) + 4
            ]
            draw.rectangle(text_bg, fill=color + (255,))
            draw.text((b[0]+2, b[1]+2), text, font=font, fill=(255,255,255,255))
        
        # Save result
        output_path = f"results_{i}.jpg" if path == "" else path
        img.convert("RGB").save(output_path)
        
def main(args):
    """main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)
    print(cfg)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    
    # Load and deploy model
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)
    
    model = Model().to(args.device)
    model.eval()
    
    # Load and process image
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)
    transforms = T.Compose([
        T.Resize((640, 640)),  
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)
    
    # Run inference
    with torch.no_grad():
        result = model(im_data, orig_size)
    
    # Process results
    labels = result[0]['labels'].detach().cpu().numpy()
    boxes = result[0]['boxes'].detach().cpu().numpy()
    scores = result[0]['scores'].detach().cpu().numpy()
    
    # Handle masks if available
    masks_list = None
    if 'masks' in result[0]:
        masks = result[0]['masks'].detach().cpu().numpy()
        print("MASKS detected")
        # Resize masks to original image size
        resized_masks = []
        for mask in masks:
            # Convert to PIL and resize
            mask_img = Image.fromarray((mask[0] * 255).astype(np.uint8))
            mask_img = mask_img.resize((w, h), Image.BILINEAR)
            # Convert back to numpy and threshold
            resized_mask = (np.array(mask_img) > 128).astype(np.float32)
            resized_masks.append(resized_mask)
        masks_list = [np.array(resized_masks)]
    
    # Create a directory to save binary masks
    masks_dir = 'binary_masks'
    os.makedirs(masks_dir, exist_ok=True)
    
    # Save the masks
    for i, mask in enumerate(resized_masks):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img.save(os.path.join(masks_dir, f'mask_{i}.png'))
    
    # Draw results
    draw(
        [im_pil],
        [labels],
        [boxes],
        [scores],
        masks_list=masks_list,
        thrh=0.5,
        path=args.output if hasattr(args, 'output') else ""
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)

