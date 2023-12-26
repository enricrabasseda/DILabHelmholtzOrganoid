### Script to generate dataset for SAM
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

def perturb_bounding_box(box, img_dim):
    """
    Perturb bounding box. 

    Args:
        box (np.array): Bounding box in format [xtopleft, ytopleft, xbottomright, ybottomright]
        img_dim (np.array): Image resolution in format [height, width]
    Returns:
        perturbed_box (np.array): Perturbed obunding box in format [xtopleft, ytopleft, xbottomright, ybottomright]
    """
    # Get image dimensions
    H, W = img_dim

    # Perturb box
    x_min = max(0, box[0] + np.random.randint(-20, 20))
    x_max = min(W, box[2] + np.random.randint(-20, 20))
    y_min = max(0, box[1] + np.random.randint(-20, 20))
    y_max = min(H, box[3] + np.random.randint(-20, 20))

    # Save perturbed box
    perturbed_box = [x_min, y_min, x_max, y_max]

    return perturbed_box

class SAMDataset(Dataset):
  def __init__(self, dataset, processor, data_folder):
    self.dataset = dataset
    self.processor = processor
    self.data_folder = data_folder

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]

    # Get the image
    image_path = item["img"]
    image = np.asarray(Image.open(os.path.join(self.data_folder, image_path)).convert("RGB"))
    # Get image shape
    H, W, _ = image.shape

    # Get ground truth mask
    mask_path = item['mask']
    rgb_mask = np.asarray(Image.open(os.path.join(self.data_folder, mask_path)).convert("RGB"))
    # Convert RGB to grayscale
    gray_mask = np.mean(rgb_mask, axis=-1)
    # Perform thresholding to convert to binary format
    ground_truth_mask = (gray_mask == 255).astype(np.uint8)

    # Perturb bounding box
    prompt = perturb_bounding_box(item['box'], [H, W])

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs