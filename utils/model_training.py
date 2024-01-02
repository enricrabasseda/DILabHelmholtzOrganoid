# File containing all functions for training
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F

def train_model(model, optimizer, seg_loss, 
                dataloader, num_epochs, device):
    """
    Args:
        model (transformers.model): model to train
        optimizer (torch.optim): optimizer for the training
        seg_loss (class): loss to use for the training given as a class with a forward function
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
            loss = seg_loss(predicted_masks, ground_truth_masks)

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch+1}')
        print(f'Mean loss: {mean(epoch_losses)}')