# Import libraries
from datasets import load_dataset
import monai
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import SamProcessor
from transformers import SamModel 

import sys
sys.path.append('/home/ubuntu')

from utils.sam_dataset import SAMDataset
from utils.model_training import train_model


# Load datasets
ds = load_dataset('json', data_files='/home/ubuntu/data/private/metadata.json')
train_dataset = ds["train"].filter(lambda example: example["split"] == "train")
val_dataset = ds["train"].filter(lambda example: example["split"] == "val")

del(ds)
print(f'Length of train dataset:', len(train_dataset))
print(f'Length of validation dataset', len(val_dataset))

# Define dataset location folder
data_folder = "/home/ubuntu/data/private"

# Import SAM Processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Define SAM datasets
sam_train_dataset = SAMDataset(dataset=train_dataset, processor=processor, data_folder=data_folder)
sam_val_dataset = SAMDataset(dataset=val_dataset, processor=processor, data_folder=data_folder)

# Create dataloaders
sam_train_dataloader = DataLoader(sam_train_dataset, batch_size=5, shuffle=True)
sam_val_dataloader = DataLoader(sam_val_dataset, batch_size=5, shuffle=True)

# Load SAM
model = SamModel.from_pretrained("facebook/sam-vit-base")
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)
del(name)

# Define training hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
# Note: Hyperparameter tuning could improve performance here
optimizer = Adam(model.mask_decoder.parameters(), lr=5e-5, weight_decay=1e-4)
# Define loss
geom_loss = monai.losses.DiceCELoss(sigmoid=False, squared_pred=True, reduction='mean')
# Define number of epochs
num_epochs = 10
# Define the patience
patience = 0

# Run training function
train_losses, val_losses, best_model = train_model(model = model, prompt = "box", optimizer = optimizer, 
                                                   geometric_loss = geom_loss, train_dataloader = sam_train_dataloader, 
                                                   val_dataloader=sam_val_dataloader, num_epochs = num_epochs, 
                                                   patience = patience, device = device, geom_interp = 150,
                                                   topological_loss = False)

# Save the trained model
save_path = "/home/ubuntu/models/geom_box-prompt.pth"
torch.save(best_model, save_path)

# Create .csv with losses
epochs = range(1,num_epochs+1)
df = pd.DataFrame(list(zip(epochs, train_losses, val_losses)),
               columns =['epoch', 'train_loss', 'val_loss'])

df.to_csv("/home/ubuntu/outputs/loss_results_geom.csv")