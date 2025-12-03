import torch
from torch import nn
from torchvision.models import resnet50


def load_from_weights(weights_path: str, eval_mode: bool = True) -> nn.Module:
    """
    Loads the WV-Net model

    Args:
        weights_path (str): Path to the weights file
        eval_mode: For inference, locks batch_norm statistics

    Returns:
        nn.Module: Loaded model
    """
    model = resnet50(weights=None)
    model.fc = nn.Identity()

    model.load_state_dict(
        torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )

    if eval_mode:
        model.eval()

    return model
