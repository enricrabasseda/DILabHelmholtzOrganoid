# Import libraries
from datasets import load_dataset
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from transformers import SamProcessor
from transformers import SamModel 

import sys
sys.path.append('/home/ubuntu')

from utils.sam_dataset import SAMDataset
from utils.metrics_calculation import metrics_dict_given_model

# Load private dataset
priv_ds = load_dataset('json', data_files='/home/ubuntu/data/private/metadata.json')
test_priv_ds = priv_ds["train"].filter(lambda example: example["split"] == "test")
del(priv_ds)
print(f'Length of test dataset of private organoids:', len(test_priv_ds))

# Load models
# Define dataset location folder
data_folder = "/home/ubuntu/data/private/"
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_test_priv_ds = SAMDataset(dataset=test_priv_ds, processor=processor, data_folder=data_folder)


# Load different models
model_train_topo_private = torch.load("/home/ubuntu/models/topo+geom_box-prompt.pth")
for name, param in model_train_topo_private.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

model_train_geom_private = torch.load("/home/ubuntu/models/geom_box-prompt.pth")
for name, param in model_train_geom_private.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

model_medsam = SamModel.from_pretrained("wanglab/medsam-vit-base")
for name, param in model_medsam.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

model_sam_base = SamModel.from_pretrained("facebook/sam-vit-base")
for name, param in model_sam_base.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

model_sam_large = SamModel.from_pretrained("facebook/sam-vit-large")
for name, param in model_sam_large.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

model_sam_huge = SamModel.from_pretrained("facebook/sam-vit-huge")
for name, param in model_sam_huge.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

print("Start first metrics calculation")

# Create dataloader for the private test dataset
sam_test_dataloader = DataLoader(sam_test_priv_ds, batch_size=1, shuffle=False)
device = "cuda"

# Compute metrics
metrics_model_1 = metrics_dict_given_model(model = model_train_topo_private.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = True, statistic = "mean")
print("1/6")
metrics_model_2 = metrics_dict_given_model(model = model_train_geom_private.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = True, statistic = "mean")
print("2/6")
metrics_medsam = metrics_dict_given_model(model = model_medsam.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = True, statistic = "mean")
print("3/6")
metrics_sam_base = metrics_dict_given_model(model = model_sam_base.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = True, statistic = "mean")
print("4/6")
metrics_sam_large = metrics_dict_given_model(model = model_sam_large.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = True, statistic = "mean")
print("5/6")
metrics_sam_huge = metrics_dict_given_model(model = model_sam_huge.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = True, statistic = "mean")
print("6/6")

# Save information in lists
models_list = ["SAM trained with Topo. + Geom. Loss and box prompt on private dataset",
               "SAM trained with Geom. Loss and box prompt on private dataset",
               "MedSAM",
               "SAM Base",
               "SAM Large",
               "SAM Huge"]
iou_list = [metrics_model_1["IoU"], metrics_model_2["IoU"], metrics_medsam["IoU"], metrics_sam_base["IoU"], metrics_sam_large["IoU"], metrics_sam_huge["IoU"]]
dsc_list = [metrics_model_1["DSC"], metrics_model_2["DSC"], metrics_medsam["DSC"], metrics_sam_base["DSC"], metrics_sam_large["DSC"], metrics_sam_huge["DSC"]]
# surfdist_list = [metrics_model_1["SurfDist"], metrics_model_2["SurfDist"], metrics_model_3["SurfDist"], metrics_model_4["SurfDist"]]
sens_list = [metrics_model_1["Sens"], metrics_model_2["Sens"], metrics_medsam["Sens"], metrics_sam_base["Sens"], metrics_sam_large["Sens"], metrics_sam_huge["Sens"]]
spec_list = [metrics_model_1["Spec"], metrics_model_2["Spec"], metrics_medsam["Spec"], metrics_sam_base["Spec"], metrics_sam_large["Spec"], metrics_sam_huge["Spec"]]
# hausdist_list = [metrics_model_1["HausDist"], metrics_model_2["HausDist"], metrics_model_3["HausDist"], metrics_model_4["HausDist"]]
ap_list = [metrics_model_1["AP"], metrics_model_2["AP"], metrics_medsam["AP"], metrics_sam_base["AP"], metrics_sam_large["AP"], metrics_sam_huge["AP"]]
f1_list = [metrics_model_1["F1"], metrics_model_2["F1"], metrics_medsam["F1"], metrics_sam_base["F1"], metrics_sam_large["F1"], metrics_sam_huge["F1"]]

# Save dataframe and later to .csv
df = pd.DataFrame(list(zip(models_list, iou_list, dsc_list, 
                           # surfdist_list,
                           sens_list, spec_list, 
                           # hausdist_list, 
                           ap_list, f1_list)),
                           columns =['Model', 'IoU', 'DSC', 
                                     # 'Surface Distance', 
                                     'Sensitivity', 'Specificity',
                                     # 'Hausdorff Distance', 
                                     'AP', 'F1'])

df.to_csv("/home/ubuntu/outputs/models_metric_noise_private.csv")

print("First metrics computed")

print("Start second metrics calculation")

# Create dataloader for the private dataset
sam_test_dataloader = DataLoader(sam_test_priv_ds, batch_size=1, shuffle=False)
device = "cuda"

# Compute metrics
metrics_model_1 = metrics_dict_given_model(model = model_train_topo_private.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = False, statistic = "mean")
print("1/6")
metrics_model_2 = metrics_dict_given_model(model = model_train_geom_private.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = False, statistic = "mean")
print("2/6")
metrics_medsam = metrics_dict_given_model(model = model_medsam.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = False, statistic = "mean")
print("3/6")
metrics_sam_base = metrics_dict_given_model(model = model_sam_base.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = False, statistic = "mean")
print("4/6")
metrics_sam_large = metrics_dict_given_model(model = model_sam_large.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = False, statistic = "mean")
print("5/6")
metrics_sam_huge = metrics_dict_given_model(model = model_sam_huge.to(device), processor = processor, test_dataloader=sam_test_dataloader, prompt = "box", device = "cuda", noised_prompt = False, statistic = "mean")
print("6/6")

# Save information in lists
models_list = ["SAM trained with Topo. + Geom. Loss and box prompt on private dataset",
               "SAM trained with Geom. Loss and box prompt on private dataset",
               "MedSAM",
               "SAM Base",
               "SAM Large",
               "SAM Huge"]
iou_list = [metrics_model_1["IoU"], metrics_model_2["IoU"], metrics_medsam["IoU"], metrics_sam_base["IoU"], metrics_sam_large["IoU"], metrics_sam_huge["IoU"]]
dsc_list = [metrics_model_1["DSC"], metrics_model_2["DSC"], metrics_medsam["DSC"], metrics_sam_base["DSC"], metrics_sam_large["DSC"], metrics_sam_huge["DSC"]]
# surfdist_list = [metrics_model_1["SurfDist"], metrics_model_2["SurfDist"], metrics_model_3["SurfDist"], metrics_model_4["SurfDist"]]
sens_list = [metrics_model_1["Sens"], metrics_model_2["Sens"], metrics_medsam["Sens"], metrics_sam_base["Sens"], metrics_sam_large["Sens"], metrics_sam_huge["Sens"]]
spec_list = [metrics_model_1["Spec"], metrics_model_2["Spec"], metrics_medsam["Spec"], metrics_sam_base["Spec"], metrics_sam_large["Spec"], metrics_sam_huge["Spec"]]
# hausdist_list = [metrics_model_1["HausDist"], metrics_model_2["HausDist"], metrics_model_3["HausDist"], metrics_model_4["HausDist"]]
ap_list = [metrics_model_1["AP"], metrics_model_2["AP"], metrics_medsam["AP"], metrics_sam_base["AP"], metrics_sam_large["AP"], metrics_sam_huge["AP"]]
f1_list = [metrics_model_1["F1"], metrics_model_2["F1"], metrics_medsam["F1"], metrics_sam_base["F1"], metrics_sam_large["F1"], metrics_sam_huge["F1"]]

# Save dataframe and later to .csv
df = pd.DataFrame(list(zip(models_list, iou_list, dsc_list, 
                           # surfdist_list,
                           sens_list, spec_list, 
                           # hausdist_list, 
                           ap_list, f1_list)),
                           columns =['Model', 'IoU', 'DSC', 
                                     # 'Surface Distance', 
                                     'Sensitivity', 'Specificity',
                                     # 'Hausdorff Distance', 
                                     'AP', 'F1'])

df.to_csv("/home/ubuntu/outputs/models_metric_gt_private.csv")

print("Second metrics computed")