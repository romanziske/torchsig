import timm
from torch.nn import Linear

__all__ = ["resnet"]


def resnet(
    input_channels: int,
    n_features: int,
    resnet_version: str = "18",
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
    features_only=False
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
    model = timm.create_model(
        "resnet" + resnet_version,
        in_chans=input_channels,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        features_only=features_only
    )

    if not features_only:
        model.fc = Linear(model.fc.in_features, n_features)
    return model
