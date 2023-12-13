### Script to load Grounding DINO model
import torch
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.slconfig import SLConfig


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    """
    Load the model.

    Args:
        repo_id (String): Path to repo: "ShilongLiu/GroundingDINO"
        filename (String): File name with weights of model: "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename (String): Configuration file name: "GroundingDINO_SwinB.cfg.py"
        device (String): "cpu" if cuda is not available
    Returns:
        model (<class 'groundingdino.models.GroundingDINO.groundingdino.GroundingDINO'>): DINO model
    """
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model