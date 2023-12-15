# File containing functions for image preprocessing
import torch
import numpy as np

def gamma_correction(image, image_source, gamma):
    """
    Return image and image_source with gamma correction preprocessing.
    Gamma correction is the process of adjusting the brightness of an image so colors appear correctly on-screen.

    Args:
      image (torch.Tensor): Tensor containing the image [C, H, W]
      image_source (np.array): Array containing the original image [H, W, C]
      gamma (float): Gamma for the correction.

    Returns:
      image_preprocessed (torch.Tensor): Tensor containing the image after correction [C, H, W]
      image_source_preprocessed (np.array): Array containing the original image after correction [H, W, C]
    """

    # Get the signs of all pixels in the image after normalization
    image_sign = torch.sign(image)

    # Perform gamma correction on the image
    image_preprocessed = image_sign * ((torch.abs(image)/torch.max(torch.abs(image))).pow(gamma))*torch.max(torch.abs(image))

    # Perform gamma correction on the image_source
    image_source_preprocessed = (((image_source/255)**gamma)*255).astype(np.uint8)

    return image_preprocessed, image_source_preprocessed