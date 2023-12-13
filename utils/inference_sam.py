# File containing all functions for SAM inference
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from groundingdino.util import box_ops


def show_mask(mask, ax, random_color=False):
    """
    Plot a mask.

    Args:
        mask (torch.Tensor): Mask in torch Tensor containing of size of image ([1,W,H])
        ax (axis): axis where we plot
        random_color (bool): give a random color to the mask if True
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """
    Plot a box.

    Args:
        box (np.array): Array containing the box in format ltbr ([4])
        ax (axis): axis where we plot
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    """
    Show boxes on the image.

    Args:
        raw_image (np.array): Array containing the original image [W, H, C]
        boxes (numpy.array): Array containing all boxes in format ltbr ([B,4])
    """
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    """
    Show points on the image.

    Args:
        raw_image (np.array): Array containing the original image [W, H, C]
        boxes (numpy.array): Array containing all boxes in format ltbr ([B,4])
        points (List): List containing all points in format xy ([P,2])
        input_labels (List): List containing labels for all points ([P])
    """
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    """
    Show points and boxes on the image.

    Args:
        raw_image (np.array): Array containing the original image [W, H, C]
        points (List): List containing all points in format xy ([P,2])
        input_labels (List): List containing labels for all points ([P])
    """
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points(coords, labels, ax, marker_size=375):
    """
    Show points and boxes on the image.

    Args:
        coords (np.array): Array containing all points in format xy ([P,2])
        labels (np.array): Array containing the labels for points. Positive = 1, Negative = 0 ([P])
        ax (axis): Axis where we plot
        marker-size (np.int8): Size of points.
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores):
    """
    Show masks on the image.

    Args:
        raw_image (np.array): Array containing the original image ([W, H, C]).
        masks (torch.Tensor): Tensor containing all masks ([1,M,W,H])
        scores (torch.Tensor): Tensor containing the scores for masks ([M]).
    """
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()

def draw_mask_together(masks, image, random_color=True):
    """
    Show all masks on a annotated frame for the image.

    Args:
        masks (torch.Tensor): Tensor containing all masks ([1,M,W,H])
        raw_image (np.array): Array containing the original image ([W, H, C])
        random_color (bool): give a random color to the mask if True
    Returns:
        annotated_frame (np.array): Array containing image with masks ([W,H,C])
    """    
    annotated_image = image
    for mask in masks:
      if random_color:
          color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
      else:
          color = np.array([30/255, 144/255, 255/255, 0.6])
      h, w = mask.shape[-2:]
      mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

      annotated_frame_pil = Image.fromarray(annotated_image).convert("RGBA")
      mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

      annotated_image = np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    return annotated_image

def sam_inference_from_dino(image_source, boxes, model, processor, device):
    """
    Get and show all masks on a annotated frame for the image given
    the boxes found with Grounding DINO.

    Args:
        image_source (np.array): Array containing the original image ([W, H, C])
        boxes (torch.Tensor): Tensor containing all boxes in cxcywh format ([B,4])
        model (<class 'transformers.models.sam.modeling_sam.SamModel'>): SAM model.
        processor (<class 'transformers.models.sam.processing_sam.SamProcessor'>): SAM processor
        device (String): "cpu" if cuda is not available
    Returns:
        masks (torch.Tensor): Tensor containing all masks ([1,M,W,H])
        scores (torch.Tensor): Tensor containing the scores for masks ([M]).
        annotated_masks (np.array): Array containing image with masks ([W,H,C])
    """   
    # Load the image and compute the embeddings
    inputs = processor(image_source, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    # Convert boxes from DINO to a compatible format for SAM
    # First scale them to the size of the image
    H, W, _ = image_source.shape
    input_boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    input_boxes = input_boxes.view(1,input_boxes.shape[0],input_boxes.shape[1])
    # Second pass input boxes to match the convention of the processor
    inputs = processor(image_source, input_boxes=input_boxes, return_tensors="pt").to(device)

    # Find the masks
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores

    # Draw masks on the plot
    annotated_masks = draw_mask_together(masks[0], image_source, random_color=True)

    return masks, scores, annotated_masks