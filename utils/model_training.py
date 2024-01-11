# File containing all functions for training
from tqdm import tqdm
from statistics import mean

import torch.nn.functional as F

from utils.topological_loss import topo_loss

def compute_loss(pred_data, true_data, geometric_loss, topo_param):
    """
    Sum the values of the geometrical loss and topological loss

    Args:
        geometric_loss (class): loss to use for the training given as a class with a forward function
        pred_data (torch.Tensor): object predicted by model in [B, C, D_1, ..., D_N] format
        true_data (torch.Tensor): ground truth data in [B, C, D_1, ..., D_N] format
        topo_param (dict): Dictionary containing all parameters needed for topological loss
    
    Returns:
        loss (float): value of the combined loss    
    """    
    # Compute each loss
    geom_loss = geometric_loss(pred_data, true_data)
    topo_loss = topo_loss(pred_data, true_data, lamda = topo_param['lamda'], 
                                 interp = topo_param['interp'], feat_d = topo_param['feat_d'],
                                 loss_q = topo_param['loss_q'], loss_r = topo_param['loss_r'])
    
    # Combine losses
    loss = geom_loss + topo_loss

    return loss

def train_model(model, optimizer, geometric_loss, dataloader, num_epochs, device, topological_loss = False,
                topo_param = {'lamda': 1.0, 'interp': 0, 'feat_d': 2, 'loss_q': 2, 'loss_r': False}):
    """
    Train a given model given the preferences.

    Args:
        model (transformers.model): model to train
        optimizer (torch.optim): optimizer for the training
        geometric_loss (class): loss to use for the training given as a class with a forward function
        topological_loss (boolean): True if topological loss will be used
        topo_param (dict): Dictionary containing all parameters needed for topological loss
        dataloader (torch.utils.data.dataloader.DataLoader): dataloader for the dataset
        num_epochs (int): number of epochs for the training
        device (string): "cuda" if available, otherwise "cpu" 
    """
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(dataloader):
            # Get ground-truth masks
            ground_truth_masks = batch["ground_truth_mask"].float().unsqueeze(1).to(device)
            _, _, m_h, m_w = ground_truth_masks.shape

            # Forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            # Get masks
            predicted_masks = outputs.pred_masks.to(device)
            # Masks post processing
            predicted_masks = F.interpolate(predicted_masks.squeeze(1), (1024, 1024), 
                                            mode="bilinear", align_corners=False)
            predicted_masks = predicted_masks[..., :992, :1024]
            predicted_masks = F.interpolate(predicted_masks, (m_h, m_w), 
                                            mode="bilinear", align_corners=False)

            # Compute loss
            # If topological loss is needed:
            if topological_loss:
                loss = compute_loss(pred_data = predicted_masks, true_data = ground_truth_masks,
                                    geometric_loss = geometric_loss, topo_param = topo_param)
            else:
                loss = geometric_loss(predicted_masks, ground_truth_masks)

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch+1}')
        print(f'Mean loss: {mean(epoch_losses)}')