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
        perturbed_box (np.array): Perturbed bounding box in format [xtopleft, ytopleft, xbottomright, ybottomright]
    """
    # Get image dimensions
    H, W = img_dim

    # Perturb box
    x_min = max(0, box[0] + np.random.randint(-15, 15))
    x_max = min(W, box[2] + np.random.randint(-15, 15))
    y_min = max(0, box[1] + np.random.randint(-15, 15))
    y_max = min(H, box[3] + np.random.randint(-15, 15))

    # Save perturbed box
    perturbed_box = [x_min, y_min, x_max, y_max]

    return perturbed_box

def center_point(box):
    """
    Get center point of the prompt box.

    Args:
        box (np.array): Bounding box in format [xtopleft, ytopleft, xbottomright, ybottomright]
    Returns:
        center_point (np.array): Center point of the box in format [x, y].
    """
    # Get middle points
    x = (box[0] + box[2])/2
    y = (box[1] + box[3])/2

    # Define center point
    center_point = [x, y]

    return center_point

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
    box_prompt = perturb_bounding_box(item['box'], [H, W])

    # Get center point that will be a prompt
    point_prompt = center_point(box_prompt)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[box_prompt]], input_points=[[point_prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask
    # add original image
    inputs["original_image"] = image
    # add original box
    inputs["gt_box"] = item['box']

    return inputs