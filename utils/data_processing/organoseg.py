# Generate patches for the dataset used in Organoseg paper.

# Import relevant libraries
import os
import pandas as pd
import cv2
from patchify import patchify
import random

# Set up the main directory and data directory
os.chdir("...")
# Data directory
data_dir = ".../data/colon_dataset"

# Initialize the lists containing all the information:
# * Directory list: contains the paths to all images.
img_source_list = []
# * Masks list: contains the path to the masks.
masks_list = []

# ------------------------------------------------------------
# Original semantic segmentation dataset creation

# Directories
original_images_dir = os.path.join(data_dir, "original", "colon_images")
original_masks_dir = os.path.join(data_dir, "original", "colon_masks")

# Get all image directories.
list_original_images_name = os.listdir(original_images_dir)

# Save all images
for i in range(len(list_original_images_name)):
    # Get image source name
    img_name, _ = os.path.splitext(list_original_images_name[i])
    # Save image path
    img_source_list.append(os.path.join(original_images_dir, img_name  + ".png"))
    # Save image mask
    masks_list.append(os.path.join(original_masks_dir, "slice_" + img_name  + "_bw.png"))

# Get image dimensions
image_array = cv2.imread(os.path.join(original_images_dir, list_original_images_name[i]))
H, W, _ = image_array.shape
print("This dataset contains", len(img_source_list), "images of size", H, "x", W, ".")

# Augmented semantic segmentation dataset generation
augmented_images_dir = os.path.join(data_dir, "augmented", "colon_images")
augmented_masks_dir = os.path.join(data_dir, "augmented", "colon_masks")
augmented_image_path_list = []
augmented_mask_path_list = []

# Get 4 square patches per image
# Calculate the dimensions of patches in each dimension
h_w_patches = 864 // 2
# Desired patch size
patch_size = (h_w_patches, h_w_patches, 3)
# Adjust the step size to ensure non-overlapping patches
step_size_h_w = h_w_patches
# Create patches
patches = patchify(image_array, patch_size, step=(H - h_w_patches, h_w_patches, 1))
print(patches.shape[0]*patches.shape[1], "patches for image of size", 
      patches.shape[3], "x", patches.shape[4])

images_name = os.listdir(original_images_dir)

# Save the paths to the new patched images and masks in the lists.
for i in range(len(images_name)):
    # Get image as array
    image_array = cv2.imread(os.path.join(original_images_dir, images_name[i]))
    # Get image source name
    img_name, _ = os.path.splitext(images_name[i])
    # Get mask as array
    mask_array = cv2.imread(os.path.join(original_masks_dir, "slice_" + img_name + "_bw.png"))
    # Get patches for the image
    image_patches = patchify(image_array, patch_size, step=(H - h_w_patches, h_w_patches, 1))
    # Get patches for the mask
    mask_patches = patchify(mask_array, patch_size, step=(H - h_w_patches, h_w_patches, 1))
    # Run through patches
    for r in range(image_patches.shape[0]):
        for c in range(image_patches.shape[1]):
            # Save image in .png format
            cv2.imwrite(os.path.join(augmented_images_dir, img_name + "_" + str(r) + str(c) + ".png"), image_patches[r,c,0,:,:])
            # Save the path of image
            img_source_list.append(os.path.join(augmented_images_dir, img_name + "_" + str(r) + str(c) + ".png"))
            augmented_image_path_list.append(os.path.join(augmented_images_dir, img_name + "_" + str(r) + str(c) + ".png"))
            # Save mask in .png format
            cv2.imwrite(os.path.join(augmented_masks_dir, "slice_" + img_name + "_" + str(r) + str(c) + "_bw.png"), mask_patches[r,c,0,:,:])
            # Save the path of mask
            masks_list.append(os.path.join(augmented_masks_dir, "slice_" + img_name + "_" + str(r) + str(c) + "_bw.png"))
            augmented_mask_path_list.append(os.path.join(augmented_masks_dir, "slice_" + img_name + "_" + str(r) + str(c) + "_bw.png"))
# Print dimensions of patch dataset
print("This dataset contains", len(augmented_image_path_list), "images of size", patch_size[0], "x", patch_size[1], ".")

# Generate metadata.json file
df = pd.DataFrame(list(zip(img_source_list, masks_list)),
               columns =['img', 'masks'])

df.to_json(data_dir + "/metadata_semantic_segmentation.json", orient = "records", lines = True)

# Get size of the dataset
print("------------------")
print("TOTAL DATASET")
print("Total number of images:", len(img_source_list))
print("Total number of masks:", len(masks_list))
print("------------------")
print("ORIGINAL DATASET")
print("Total number of images:", 64)
print("Total number of masks:", 64)
print("------------------")
print("AUGMENTED DATASET")
print("Total number of images:", len(augmented_image_path_list))
print("Total number of masks:", len(augmented_mask_path_list))
print("------------------")

# ------------------------------------------------------------------
# Instance segmentation augmented dataset creation

# Get paths to all relevant directories
augmented_seg_images_dir = os.path.join(data_dir, "augmented", "colon_images")
augmented_seg_masks_dir = os.path.join(data_dir, "augmented", "colon_masks")
augmented_inst_masks_dir = os.path.join(data_dir, "augmented", "colon_instance_masks")
# Relative path to images
list_augmented_seg_images_name = os.listdir(augmented_seg_images_dir)

# Create empty lists to save the information
image_path_list = []
mask_path_list = []
box_list = []

# Create one instance segmentation mask per image using connected component analysis
for i in range(len(list_augmented_seg_images_name)):
    # Load the image and semantic segmentation mask
    img_name, _ = os.path.splitext(list_augmented_seg_images_name[i])
    mask_path = augmented_seg_masks_dir + "/" + "slice_" + img_name + "_bw.png"
    img_path = "colon_dataset/augmented/colon_images/" + img_name + ".png"
    # Assuming 'mask' is your binary mask, load it
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Perform connected components analysis
    output = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
    # Here we save the relevant information about the connected components
    (numLabels, labels, stats, centroids) = output
    # Save the masks in the corresponding file and all relevant information: framing box and paths; to be saved later in metadata.json file.
    for label in range(1,stats.shape[0]):
        # Only save masks that are bigger than a threshold
        if stats[label,4] > 2000:
            if label < 10:
                num = "0" + str(label)
            else:
                num = str(label)
            # Get mask according to the label
            mask_for_label = (labels == label)*255
            # Save it
            mask_path = "colon_dataset/augmented/colon_instance_masks/" + "slice_" + img_name + "_" + num + "_bw.png"
            cv2.imwrite("/home/ubuntu/data/" + mask_path, mask_for_label)
            # Get box corresponding to mask
            box_mask = [stats[label,0], stats[label,1], stats[label,0] + stats[label,2], stats[label,1] + stats[label,3]]
            # Save information in the lists
            image_path_list.append(img_path)
            mask_path_list.append(mask_path)
            box_list.append(box_mask)
# Get dimensions of the dataset
print("There are", len(list_augmented_seg_images_name), "images with a total number of", len(mask_path_list), "organoids.")

# Split dataset in train/val/test (60%/20%/20%)
random.seed(4)
split_list = []
for i in range(len(image_path_list)):
    # If there is at least one mask for this image save one line per mask
    # Get dataset split and save the same for all masks corresponding to same image
    split_list.append(random.choices(["train", "val", "test"], weights=[0.6, 0.2, 0.2])[0])

# Save all information in metadata.json file
df = pd.DataFrame(list(zip(image_path_list, box_list, mask_path_list, split_list)),
               columns =['img', 'box', 'mask', 'split'])

df.to_json(data_dir + "/metadata_instance_segmentation.json", orient = "records", lines = True)

