

import os
import torch
from config import TrainingConfig
from utils import get_ssl_model, parse_args, print_config


def get_backbone_filename(config: TrainingConfig) -> str:
    return (
        f"backbone"
        f"-{config.backbone}"
        f"-{config.dataset}"
        f"-{'spec' if config.spectrogram else 'iq'}"
        f"-nf{config.n_features}"
        f".pth"
    )


def export_backbone(config: TrainingConfig):
    if config.checkpoint is None:
        raise ValueError("Checkpoint not provided")

    checkpoint_path = os.path.join(config.training_path, config.checkpoint)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    ssl_model = get_ssl_model(config)
    # load model weights from checkpoint
    checkpoint = torch.load(checkpoint_path)
    ssl_model.load_state_dict(checkpoint["state_dict"])

    # export the backbone
    torch.save(
        ssl_model.backbone.state_dict(),
        get_backbone_filename(config),
    )

    print("Backbone exported successfully")


if __name__ == "__main__":

    config = parse_args()
    print_config(config)
    export_backbone(config)
