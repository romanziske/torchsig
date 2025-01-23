from typing import Literal
import timm

from torch import nn
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model

__all__ = ["resnet"]


class SelectStage(nn.Module):
    """Selects features from a specific ResNet stage."""

    def __init__(self, stage: str = "res5"):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return x[self.stage]


def build_detectron2_resnet(
    input_channels: int,
    n_features: int,
    resnet_version: str = "50",
) -> nn.Module:
    """Build ResNet backbone using Detectron2."""

    # Initialize config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        f"COCO-Detection/faster_rcnn_R_{resnet_version}_FPN_3x.yaml"
    ))

    # Configure model
    cfg.MODEL.WEIGHTS = ""  # No pretrained weights
    cfg.MODEL.PIXEL_MEAN = [0.0] * input_channels  # Zero mean per channel
    cfg.MODEL.PIXEL_STD = [1.0] * input_channels   # Unit std per channel
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.DEVICE = "mps"

    # Build model
    det_model = build_model(cfg)

    # Create backbone with pooling just like in timm
    backbone = nn.Sequential(
        det_model.backbone.bottom_up,
        SelectStage("res5"),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2048, n_features)
    )

    return backbone


def resnet(
    input_channels: int,
    n_features: int,
    resnet_version: str = "18",
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    features_only=False,
    from_lib: Literal["timm", "detectron2"] = "timm",
):
    """Constructs and returns a version of the ResNet model.
    Args:

        input_channels (int):
            Number of input channels; should be 2 for complex spectrograms

        n_features (int):
            Number of output features; should be the number of classes when used directly for classification

        resnet_version (str):
            Specifies the version of resnet to use. See the timm resnet documentation for details. Examples are '18', '34' or'50'

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """

    if from_lib == "timm":
        model = timm.create_model(
            "resnet" + resnet_version,
            in_chans=input_channels,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            features_only=features_only
        )

        if not features_only:
            model.fc = nn.Linear(model.fc.in_features, n_features)

    elif from_lib == "detectron2":
        return build_detectron2_resnet(
            input_channels=input_channels,
            n_features=n_features,
            resnet_version=resnet_version
        )

    return model
