# infer.py
import torch
import torch.nn as nn
import torchvision.transforms as T
# from torch.cuda.amp import autocast # Not used here, can remove if not needed for your setup
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor # Keep ImageColor
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) # Ensure this path is correct
# import argparse # Already imported at the end
# import src.misc.dist as dist # Not used here
from src.core import YAMLConfig
# from src.solver import TASKS # Not used here

# NMS and Slicing functions (postprocess, slice_image, merge_predictions) are for bounding boxes.
# We will focus on the direct output for masks for now.
# If you use slicing, mask merging would be a separate complex step.

def draw_detections(image_pil, detections, class_names=None, score_threshold=0.5, font_path="arial.ttf", output_path="result_drawn.jpg"):
    """
    Draws bounding boxes, labels, scores, and masks on an image.
    Args:
        image_pil (PIL.Image): The input image.
        detections (dict): A dictionary containing 'labels', 'boxes', 'scores', and 'masks'.
                           'masks' should be binary [N_dets, H_orig, W_orig].
        class_names (list, optional): List of class names.
        score_threshold (float, optional): Threshold to filter detections.
        font_path (str, optional): Path to a .ttf font file.
        output_path (str, optional): Path to save the drawn image.
    """
    img_draw = image_pil.copy().convert("RGBA") # Use RGBA for transparency
    draw = ImageDraw.Draw(img_draw, "RGBA")

    try:
        font = ImageFont.truetype(font_path, 15)
    except IOError:
        font = ImageFont.load_default()

    # Define some colors for classes (can be expanded)
    # Using distinct RGBA colors
    _COLORS = np.array([
        [0.000, 0.447, 0.741, 0.5], [0.850, 0.325, 0.098, 0.5], [0.929, 0.694, 0.125, 0.5],
        [0.494, 0.184, 0.556, 0.5], [0.466, 0.674, 0.188, 0.5], [0.301, 0.745, 0.933, 0.5],
        [0.635, 0.078, 0.184, 0.5], [0.300, 0.300, 0.300, 0.5], [0.700, 0.700, 0.700, 0.5],
        [1.000, 0.000, 0.000, 0.5], [0.000, 1.000, 0.000, 0.5], [0.000, 0.000, 1.000, 0.5],
        [1.000, 1.000, 0.000, 0.5], [1.000, 0.000, 1.000, 0.5], [0.000, 1.000, 1.000, 0.5]
    ]).astype(np.float32).reshape(-1, 4)
    
    labels = detections['labels']
    boxes = detections['boxes'] # Assumed to be xyxy, absolute
    scores = detections['scores']
    masks = detections.get('masks', None) # Binary masks [N_dets, H_orig, W_orig]

    # Filter by score
    keep = scores > score_threshold
    
    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]
    if masks is not None:
        masks = masks[keep]

    print(f"Drawing {len(labels)} detections with score > {score_threshold}")

    # Draw masks first
    if masks is not None:
        for i in range(len(labels)):
            mask_np = masks[i] # This is already binary {0, 1} and at original image size
            
            # Get color for the mask based on class label
            color_idx = labels[i] % len(_COLORS)
            mask_color_rgba = tuple(int(c * 255) for c in _COLORS[color_idx]) # (R, G, B, Alpha)

            # Create a colored overlay for the mask
            # mask_np is [H, W], values 0 or 1
            # Create an RGBA image for the mask:
            # Where mask_np is 1, use mask_color_rgba, otherwise transparent
            h, w = mask_np.shape
            colored_mask_img = Image.new("RGBA", (w, h), (0,0,0,0)) # Transparent base
            
            # Get pixel access
            pixels = colored_mask_img.load()
            for r_idx in range(h):
                for c_idx in range(w):
                    if mask_np[r_idx, c_idx] > 0.5: # If it's a foreground pixel
                        pixels[c_idx, r_idx] = mask_color_rgba
            
            # Alpha composite this colored mask onto the main image
            img_draw.alpha_composite(colored_mask_img)
            
    # Re-initialize Draw object because alpha_composite creates a new image if img_draw was not RGBA
    draw = ImageDraw.Draw(img_draw)

    # Draw boxes and labels on top
    for i in range(len(labels)):
        box = boxes[i]
        label_idx = labels[i]
        score = scores[i]

        color_idx = label_idx % len(_COLORS) # Use a different alpha for box/text
        box_color_rgb = tuple(int(c * 255) for c in _COLORS[color_idx][:3]) 

        draw.rectangle(box.tolist(), outline=box_color_rgb, width=2)

        label_text = f"Class {label_idx}" if class_names is None else class_names[label_idx]
        text = f"{label_text}: {score:.2f}"
        
        # Text position and background
        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        text_x = box[0]
        text_y = box[1] - (text_bbox[3] - text_bbox[1]) - 2 # Place text above the box
        if text_y < 0: # If text goes off screen, place it below
            text_y = box[3] + 2

        text_bg_coords = [text_x, text_y, text_x + (text_bbox[2]-text_bbox[0]), text_y + (text_bbox[3]-text_bbox[1])]
        draw.rectangle(text_bg_coords, fill=box_color_rgb)
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    img_draw.convert("RGB").save(output_path)
    print(f"Saved detection result to {output_path}")


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint.get('ema', {}).get('module', checkpoint.get('model'))
    else:
        raise AttributeError('Resume path is required to load model state_dict.')

    # --- Model Definition ---
    # This assumes 'cfg.model' is the RTDETR (nn.Module) itself
    # and 'cfg.postprocessor' is the RTDETRPostProcessor (nn.Module)
    # Make sure they are instantiated correctly from your YAML
    
    # Example: Manually instantiate if cfg.model is not yet the RTDETR object
    # from src.zoo.rtdetr.rtdetr import RTDETR # Adjust import
    # from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer # Adjust
    # from src.zoo.rtdetr.rtdetr_encoder import RTDETREncoder # Adjust
    # from src.zoo.rtdetr.hybrid_encoder import HybridEncoder # Adjust
    # from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor # Adjust
    # from src.zoo.resnet import ResNet # Adjust
    # Add any other necessary components (backbone parts, etc.)

    # This part is highly dependent on how your YAMLConfig instantiates these
    # Assuming cfg.model is already an RTDETR instance and cfg.postprocessor is RTDETRPostProcessor instance
    rtdetr_model = cfg.model 
    rtdetr_model.load_state_dict(state)
    rtdetr_model = rtdetr_model.deploy().to(args.device) # Call deploy on the RTDETR model for its components
    
    postprocessor = cfg.postprocessor.deploy().to(args.device) # Deploy postprocessor

    # Load and process image
    im_pil = Image.open(args.im_file).convert('RGB')
    original_w, original_h = im_pil.size
    
    # Define preprocessing transforms (consistent with training if possible, but resize is key)
    # Match the input size your model was trained on or expects (e.g., 640x640)
    INPUT_SIZE = 640 # Example, adjust if your model uses a different size
    
    # Simple Resize and ToTensor, Normalize if your model expects it
    # Common normalization for ImageNet pre-trained models
    # MEAN = [0.485, 0.456, 0.406]
    # STD = [0.229, 0.224, 0.225]
    # normalize = T.Normalize(mean=MEAN, std=STD) # Add if needed

    transforms = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        # normalize, # Add if your model expects normalization
    ])
    img_tensor = transforms(im_pil).unsqueeze(0).to(args.device) # [1, C, H_in, W_in]

    # orig_target_sizes for postprocessor: [B, 2] with [W_orig, H_orig] for each item
    orig_target_sizes = torch.tensor([[original_w, original_h]], dtype=torch.float32).to(args.device)

    # Run inference
    with torch.no_grad():
        model_outputs = rtdetr_model(img_tensor) # Model outputs dict: pred_logits, pred_boxes, pred_masks
        # Postprocess
        results = postprocessor(model_outputs, orig_target_sizes) # List of dicts

    # Results is a list (batch size = 1 here), so take the first element
    detection_results = results[0] 

    # Extract to NumPy arrays
    labels_np = detection_results['labels'].cpu().numpy()
    boxes_np = detection_results['boxes'].cpu().numpy() # Should be absolute xyxy
    scores_np = detection_results['scores'].cpu().numpy()
    
    masks_np_list = None
    if 'masks' in detection_results:
        # masks are already binary {0,1} and at original image size
        # Shape [num_top_queries, H_orig, W_orig]
        masks_np = detection_results['masks'].cpu().numpy() 
        masks_np_list = [masks_np] # For the draw function, wrap in a list for the batch
        print(f"MASKS detected in infer.py. Shape: {masks_np.shape}, Unique values: {np.unique(masks_np)}")

        # Save individual binary masks (optional debug)
        masks_dir = 'binary_masks_inferred'
        os.makedirs(masks_dir, exist_ok=True)
        for i, mask_arr in enumerate(masks_np): # Iterate through masks for top queries
            if scores_np[i] > 0.1: # Only save masks for reasonably confident detections
                mask_img_to_save = Image.fromarray((mask_arr * 255).astype(np.uint8), mode='L')
                mask_img_to_save.save(os.path.join(masks_dir, f'inferred_mask_query_{i}_score_{scores_np[i]:.2f}.png'))
    
    # Determine output path
    output_file_path = args.output if hasattr(args, 'output') and args.output else "result_drawn.jpg"
    if os.path.isdir(output_file_path): # If a directory is given, create a filename
        output_file_path = os.path.join(output_file_path, f"result_{os.path.basename(args.im_file)}")

    # Use the new draw function
    draw_detections(
        im_pil,
        {'labels': labels_np, 'boxes': boxes_np, 'scores': scores_np, 'masks': masks_np if masks_np_list else None},
        score_threshold=0.1, # Adjust score threshold as needed
        output_path=output_file_path
    )

if __name__ == '__main__':
    import argparse # Moved import here
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the config YAML file")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the checkpoint file (.pth)")
    parser.add_argument('-f', '--im_file', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output', type=str, default="result_drawn.jpg", help="Path to save the output drawn image or directory")
    # parser.add_argument('-s', '--sliced', type=bool, default=False) # Slicing logic removed for simplicity
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # parser.add_argument('-nc', '--numberofboxes', type=int, default=25) # num_top_queries in postprocessor handles this
    args = parser.parse_args()
    main(args)