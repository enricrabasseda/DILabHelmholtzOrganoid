import torch.nn.functional as F
import torch

from utils.metrics import iou, dsc, surfdist, sensitivity, specificity, hausdorff_dist, ap, f1_score

from statistics import mean, median
def metrics_dict_given_model(model, processor, test_dataloader, prompt, device, noised_prompt, statistic):
    """
    Compute all the metrics above for one model on a given dataset.

    Args:
        model (transformers.model): model to use
        processor (transformers.SamModel.processor): processor of inputs
        test_dataloader (torch.utils.data.dataloader.DataLoader): dataloader for the test dataset
        prompt (string): "box" or "point" depending on how we want to do the inference
        device (string): device where we work
        noised_prompt (Bool): denoise prompt
        statistic (string): "mean" or "median" to aggregate all metrics values

    Returns:
        metrics_dict (dictionary): dictionary containing all array with metrics for every image
    """
    iou_values = []
    dsc_values = []
    # surfdist_values = []
    sens_values = []
    spec_values = []
    # hausdist_values = []
    ap_values = []
    f1_values = []

    # Set model to device
    model.to(device)
    # Set evaluation mode
    model.eval()

    # Run on all batches
    for batch in test_dataloader:
        # Get ground-truth masks
        ground_truth_masks = batch["ground_truth_mask"].float().unsqueeze(1).to(device)
        _, _, m_h, m_w = ground_truth_masks.shape

        with torch.no_grad():
            if noised_prompt:
                # Forward pass
                if prompt == "point":
                    outputs = model(pixel_values=batch["pixel_values"].to(device),
                                    input_points=batch["input_points"].to(device),
                                    multimask_output=False)
                elif prompt == "box":
                    outputs = model(pixel_values=batch["pixel_values"].to(device),
                                    input_boxes=batch["input_boxes"].to(device),
                                    multimask_output=False)
            elif not noised_prompt: 
                # Get middle point
                x = (batch["gt_box"][0] + batch["gt_box"][2])/2
                y = (batch["gt_box"][1] + batch["gt_box"][3])/2

                # Compute the inputs for the desired image
                inputs = processor(batch["original_image"], input_boxes=[[batch["gt_box"]]], 
                                input_points=[[[x.item(), y.item()]]], return_tensors="pt")  

                # Forward pass
                if prompt == "point":
                    outputs = model(pixel_values=inputs["pixel_values"].to(device),
                                    input_points=inputs["input_points"].to(device),
                                    multimask_output=False)
                elif prompt == "box":
                    outputs = model(pixel_values=inputs["pixel_values"].to(device),
                                    input_boxes=inputs["input_boxes"].to(device),
                                    multimask_output=False)
            # Get masks
            predicted_masks = outputs.pred_masks.to(device)
            # Masks post processing
            predicted_masks = F.interpolate(predicted_masks.squeeze(1), (1024, 1024), 
                                            mode="bilinear", align_corners=False)
            predicted_masks = F.interpolate(predicted_masks, (m_h, m_w), 
                                            mode="bilinear", align_corners=False)
            
            # Apply sigmoid
            predicted_masks = torch.sigmoid(predicted_masks)

            # Compute metrics
            iou_values.append(iou(predicted_masks, ground_truth_masks))
            dsc_values.append(dsc(predicted_masks, ground_truth_masks))
            # surfdist_values.append(surfdist(predicted_masks, ground_truth_masks))
            sens_values.append(sensitivity(predicted_masks, ground_truth_masks))
            spec_values.append(specificity(predicted_masks, ground_truth_masks))
            # hausdist_values.append(hausdorff_dist(predicted_masks, ground_truth_masks))
            ap_values.append(ap(predicted_masks, ground_truth_masks))
            f1_values.append(f1_score(predicted_masks, ground_truth_masks))
    
    if statistic == "mean":
        # Create dictionary with all values
        metrics_dict = {}
        metrics_dict["IoU"] = mean(iou_values)
        metrics_dict["DSC"] = mean(dsc_values)
        # metrics_dict["SurfDist"] = mean(surfdist_values)
        metrics_dict["Sens"] = mean(sens_values)
        metrics_dict["Spec"] = mean(spec_values)
        # metrics_dict["HausDist"] = mean(hausdist_values)
        metrics_dict["AP"] = mean(ap_values)
        metrics_dict["F1"] = mean(f1_values)
    if statistic == "median":
        # Create dictionary with all values
        metrics_dict = {}
        metrics_dict["IoU"] = median(iou_values)
        metrics_dict["DSC"] = median(dsc_values)
        # metrics_dict["SurfDist"] = mean(surfdist_values)
        metrics_dict["Sens"] = median(sens_values)
        metrics_dict["Spec"] = median(spec_values)
        # metrics_dict["HausDist"] = mean(hausdist_values)
        metrics_dict["AP"] = median(ap_values)
        metrics_dict["F1"] = median(f1_values)        

    return metrics_dict
