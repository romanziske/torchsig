

import torch
from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
from torchsig.models.iq_models.resnet.resnet1d import ResNet1d
from torchsig.models.ssl_models.byol import BYOL


def export_backbone():
    # number of features for the embedding
    n_features = 2048

    class_list = list(TorchSigNarrowband._idx_to_name_dict.values())
    num_classes = len(class_list)
    batch_size = 16

    backbone = ResNet1d(
        input_channels=2,
        n_features=n_features,
        resnet_version="50",
    )

    ssl_model = BYOL(
        backbone=backbone,
        num_ftrs=n_features,
        hidden_dim=4096,
        out_dim=256,
        num_classes=num_classes,
        batch_size_per_device=batch_size,
        use_online_linear_eval=True,
    )
    # load model weights from checkpoint
    checkpoint = torch.load(
        "train/byol/lightning_logs/version_0/byol-ResNet-epoch=88-train_loss=-3.99.ckpt",
    )
    ssl_model.load_state_dict(checkpoint["state_dict"])

    # export the backbone
    torch.save(ssl_model.backbone.state_dict(), "backbone.pth")

    print("Backbone exported successfully")


if __name__ == "__main__":

    export_backbone()
