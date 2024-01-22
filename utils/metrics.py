import torch
import monai

from torchmetrics.classification import AveragePrecision

def iou(pred_masks, true_masks):
    """
    Compute the intersection over union of the bounding boxes for the masks.

    Args:
        pred_masks: predicted masks of shape [B, 1, H, W] with values between 0 and 1
        true_masks: true masks of shape [B, 1, H, W] with value 0 (background) and 1 (foreground)

    Returns:
        avg_iou: average Intersection over Union on the pixels of the masks over all batch masks
    """
    # Convert soft values to 0 or 1 with a threshold of 0.5
    pred_masks_hard = (pred_masks.squeeze(1) > 0.5).float()
    # Convert ground truth masks to float
    gt_masks_binary = true_masks.squeeze(1).float()

    # Compute intersection
    intersection = torch.sum(pred_masks_hard*gt_masks_binary, dim=(1,2))

    # Compute union
    union = torch.sum(pred_masks_hard, dim=(1,2)) + torch.sum(gt_masks_binary, dim=(1,2)) - intersection
    
    # Compute IoU
    iou = intersection/union

    # Return average IoU
    avg_iou = iou.mean().item()
    return avg_iou

def dsc(pred_masks, true_masks):
    """
    Compute the Dice Similarity Coefficient: DSC(G,P) = 2*|G\cap P|/(|G| + |P|)

    Args:
        pred_masks: predicted masks of shape [B, 1, H, W] with values between 0 and 1
        true_masks: true masks of shape [B, 1, H, W] with value 0 (background) and 1 (foreground)

    Returns:
        avg_dsc: average Dice Similarity Coefficient on the pixels of the masks over all batch masks
    """   
    # Convert soft values to 0 or 1 with a threshold of 0.5
    pred_masks_hard = (pred_masks > 0.5).float()
    # Convert ground truth masks to float
    gt_masks_binary = true_masks.float()

    # Compute mean dice using MONAI metrics
    dice_metric = monai.metrics.DiceMetric(include_background = False)
    dice = dice_metric(pred_masks_hard, gt_masks_binary)

    # Return the average DSCs
    avg_dsc = torch.mean(dice).item()

    return avg_dsc

def surfdist(pred_masks, true_masks):
    """
    Compute the Surface Distance

    Args:
        pred_masks: predicted masks of shape [B, 1, H, W] with values between 0 and 1
        true_masks: true masks of shape [B, 1, H, W] with value 0 (background) and 1 (foreground)

    Returns:
        avg_sd: average Surface Distance on the pixels of the masks over all batch masks
    """   
    # Convert soft values to 0 or 1 with a threshold of 0.5
    pred_masks_hard = (pred_masks > 0.5).float()
    # Convert ground truth masks to float
    gt_masks_binary = true_masks.float()

    # Compute NSD using MONAI class SurfaceDistanceMetric
    surface_distance_metric = monai.metrics.SurfaceDistanceMetric()
    result = surface_distance_metric(pred_masks_hard, gt_masks_binary)

    # Compute the mean normalized surface distance
    avg_sd = torch.mean(result).item()

    return avg_sd

def sensitivity(pred_masks, true_masks):
    """
    Compute the Sensitivity

    Args:
        pred_masks: predicted masks of shape [B, 1, H, W] with values between 0 and 1
        true_masks: true masks of shape [B, 1, H, W] with value 0 (background) and 1 (foreground)

    Returns:
        avg_sensitivity: average sensitivity on the pixels of the masks over all batch masks
    """
    # Convert soft values to 0 or 1 with a threshold of 0.5
    pred_masks_hard = (pred_masks > 0.5).float()
    # Convert ground truth masks to float
    gt_masks_binary = true_masks.float()

    # Compute confusion matrix
    conf_matrix = monai.metrics.get_confusion_matrix(pred_masks_hard, gt_masks_binary, include_background=True)
    sensitivity = monai.metrics.compute_confusion_matrix_metric("sensitivity", conf_matrix)
    # Compute the mean sensitivity
    avg_sensitivity = torch.mean(sensitivity).item()

    return avg_sensitivity

def specificity(pred_masks, true_masks):
    """
    Compute the Specificity

    Args:
        pred_masks: predicted masks of shape [B, 1, H, W] with values between 0 and 1
        true_masks: true masks of shape [B, 1, H, W] with value 0 (background) and 1 (foreground)

    Returns:
        avg_specificity: average specificity on the pixels of the masks over all batch masks
    """
    # Convert soft values to 0 or 1 with a threshold of 0.5
    pred_masks_hard = (pred_masks > 0.5).float()
    # Convert ground truth masks to float
    gt_masks_binary = true_masks.float()

    # Compute confusion matrix and specificity
    conf_matrix = monai.metrics.get_confusion_matrix(pred_masks_hard, gt_masks_binary, include_background=False)
    specificity = monai.metrics.compute_confusion_matrix_metric("specificity", conf_matrix)

    # Compute the mean specificity
    avg_specificity = torch.mean(specificity).item()

    return avg_specificity 

def hausdorff_dist(pred_masks, true_masks):
    """
    Compute Hausdorff distance.
    
    Args:
        pred_masks: predicted masks of shape [B, 1, H, W] with values between 0 and 1
        true_masks: true masks of shape [B, 1, H, W] with value 0 (background) and 1 (foreground)

    Returns:
        avg_hausdist: average hausdorff distance on the pixels of the masks over all batch masks
    """   
    # Convert soft values to 0 or 1 with a threshold of 0.5
    pred_masks_hard = (pred_masks > 0.5).float()
    # Convert ground truth masks to float
    gt_masks_binary = true_masks.float()

    # Compute all specificity values using MONAI
    hausdorff_metric = monai.metrics.HausdorffDistanceMetric()
    hausdorff_dist_values = hausdorff_metric(pred_masks_hard, gt_masks_binary)

    # Compute the mean Hausdorff distance across the batch
    avg_hausdorffdist = torch.mean(hausdorff_dist_values).item()

    return avg_hausdorffdist

def ap(pred_masks, true_masks):
    """
    Compute Average Precision (AP) computing the integral of precision-recall curve.
    
    Args:
        pred_masks: predicted masks of shape [B, 1, H, W] with values between 0 and 1
        true_masks: true masks of shape [B, 1, H, W] with value 0 (background) and 1 (foreground)

    Returns:
        avg_precision: AP on the pixels of the masks over all batch masks
    """   
    # Flatten mask tensors
    pred_masks_flat = pred_masks.view(-1)
    true_masks_flat = true_masks.view(-1)

    # Compute average precision
    average_precision = AveragePrecision(task="binary")
    avg_precision = average_precision(pred_masks_flat, true_masks_flat.long()).mean().item()

    return avg_precision      

def f1_score(pred_masks, true_masks):
    """
    Compute F1 score 

    Args:
        pred_masks: predicted masks of shape [B, 1, H, W] with values between 0 and 1
        true_masks: true masks of shape [B, 1, H, W] with value 0 (background) and 1 (foreground)

    Returns:
        avg_f1_score: F1 score
    """   
    # Convert soft values to 0 or 1 with a threshold of 0.5
    pred_masks_hard = (pred_masks > 0.5).float()
    # Convert ground truth masks to float
    gt_masks_binary = true_masks.float()

    # Compute confusion matrix and f1 score
    conf_matrix = monai.metrics.get_confusion_matrix(pred_masks_hard, gt_masks_binary, include_background=False)
    f1_values = monai.metrics.compute_confusion_matrix_metric("f1 score", conf_matrix)

    # Compute the mean f1 score
    avg_f1_score = torch.mean(f1_values).item()

    return avg_f1_score         