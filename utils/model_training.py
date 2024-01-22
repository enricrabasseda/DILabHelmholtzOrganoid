# File containing all functions for training
from tqdm import tqdm
from statistics import mean

import torch.nn.functional as F
import torch

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
    # Compute each loss, geometric loss is configured to apply a sigmoid function to the input automatically
    geom_loss = geometric_loss(pred_data, true_data)
    # Note we need to apply sigmoid function on pred data, since the loss function will not do it
    top_loss = topo_loss(torch.sigmoid(pred_data), true_data, lamda = topo_param['lamda'], 
                         interp = topo_param['interp'], feat_d = topo_param['feat_d'],
                         loss_q = topo_param['loss_q'], loss_r = topo_param['loss_r'])
    
    # Combine losses
    loss = geom_loss + top_loss

    return loss

def train_model(model, prompt, optimizer, geometric_loss, train_dataloader, val_dataloader, num_epochs, device, geom_interp = 0,
                topological_loss = False, topo_param = {'lamda': 1.0, 'interp': 0, 'feat_d': 2, 'loss_q': 2, 'loss_r': False}):
    """
    Train a given model given the preferences.

    Args:
        model (transformers.model): model to train
        prompt (string): "box" or "point" depending on the preference. If None we train without input
        optimizer (torch.optim): optimizer for the training
        geometric_loss (class): loss to use for the training given as a class with a forward function
        geom_interp (int): size to downsample mask for geometrical loss computation, if 0 no interpolation is done (by default)
        topological_loss (boolean): True if topological loss will be used
        topo_param (dict): Dictionary containing all parameters needed for topological loss
        train_dataloader (torch.utils.data.dataloader.DataLoader): dataloader for the train dataset
        val_dataloader (torch.utils.data.dataloader.DataLoader): dataloader for the validation dataset
        num_epochs (int): number of epochs for the training
        device (string): "cuda" if available, otherwise "cpu" 
    """
    model.to(device)
    for epoch in range(num_epochs):
        # Set training mode
        model.train()
        # Train the model on train dataset
        epoch_train_losses = []
        for batch in tqdm(train_dataloader):
            # Get ground-truth masks
            ground_truth_masks = batch["ground_truth_mask"].float().unsqueeze(1).to(device)
            _, _, m_h, m_w = ground_truth_masks.shape

            # Forward pass
            if prompt is None:
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                multimask_output=False)
            elif prompt == "point":
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                input_points=batch["input_points"].to(device),
                                multimask_output=False)
            elif prompt == "box":
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                input_boxes=batch["input_boxes"].to(device),
                                multimask_output=False)
            # Get masks
            predicted_masks = outputs.pred_masks.to(device)
            # Masks post processing
            predicted_masks = F.interpolate(predicted_masks.squeeze(1), (1024, 1024), 
                                            mode="bilinear", align_corners=False)
            predicted_masks = predicted_masks[..., :992, :1024]
            # If geometric interpolation, then we downsample the masks: ground-truth and predicted ones
            if geom_interp != 0:
                predicted_masks = F.interpolate(predicted_masks, (geom_interp, geom_interp), 
                                                mode="bilinear", align_corners=False)
                ground_truth_masks = F.interpolate(ground_truth_masks, (geom_interp, geom_interp),
                                                   mode="bilinear", align_corners=False)
            else:
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
            epoch_train_losses.append(loss.item())

        print(f'EPOCH: {epoch+1}')
        print(f'Train dataset mean loss: {mean(epoch_train_losses)}')

        # Set evaluation mode to set dropout and batch norm. layers to evaluation mode
        model.eval()
        # Evaluate the model on the validation dataset
        epoch_val_losses = []
        for batch in tqdm(val_dataloader):
            # Get ground-truth masks
            ground_truth_masks = batch["ground_truth_mask"].float().unsqueeze(1).to(device)
            _, _, m_h, m_w = ground_truth_masks.shape

            # Do not compute gradients anywhere
            with torch.no_grad():
            # Forward pass
                if prompt is None:
                    outputs = model(pixel_values=batch["pixel_values"].to(device),
                                    multimask_output=False)
                elif prompt == "point":
                    outputs = model(pixel_values=batch["pixel_values"].to(device),
                                    input_points=batch["input_points"].to(device),
                                    multimask_output=False)
                elif prompt == "box":
                    outputs = model(pixel_values=batch["pixel_values"].to(device),
                                    input_boxes=batch["input_boxes"].to(device),
                                    multimask_output=False)
                # Get masks
                predicted_masks = outputs.pred_masks.to(device)
                # Masks post processing
                predicted_masks = F.interpolate(predicted_masks.squeeze(1), (1024, 1024), 
                                                mode="bilinear", align_corners=False)
                predicted_masks = predicted_masks[..., :992, :1024]
                # If geometric interpolation, then we downsample the masks: ground-truth and predicted ones
                if geom_interp != 0:
                    predicted_masks = F.interpolate(predicted_masks, (geom_interp, geom_interp), 
                                                    mode="bilinear", align_corners=False)
                    ground_truth_masks = F.interpolate(ground_truth_masks, (geom_interp, geom_interp),
                                                    mode="bilinear", align_corners=False)
                else:
                    predicted_masks = F.interpolate(predicted_masks, (m_h, m_w), 
                                                    mode="bilinear", align_corners=False)

                # Compute loss
                # If topological loss is needed:
                if topological_loss:
                    loss = compute_loss(pred_data = predicted_masks, true_data = ground_truth_masks,
                                        geometric_loss = geometric_loss, topo_param = topo_param)
                else:
                    loss = geometric_loss(predicted_masks, ground_truth_masks)            
                
                # Add loss
                epoch_val_losses.append(loss.item())
        
        print(f'Validation dataset mean loss: {mean(epoch_val_losses)}')
