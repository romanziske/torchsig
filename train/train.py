

import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, callbacks
from lightning.pytorch.loggers import TensorBoardLogger
import os

from torchsig.datasets.datamodules import NarrowbandDataModule

from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
from torchsig.models.ssl_models import BYOL
from torchsig.models.iq_models.resnet import ResNet1d
from torchsig.transforms.byol_transform import BYOLTransform
from torchsig.transforms.target_transforms import ClassIdxToTensor, DescToClassIndex
from torchsig.transforms.transforms import ComplexTo2D, Compose, Normalize, ToTensor


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def collate_fn(batch):
    """Custom collate for BYOL views"""
    views, targets = zip(*batch)
    view1s, view2s = zip(*views)

    return (
        torch.stack(view1s),
        torch.stack(view2s)
    ), torch.tensor(targets)


def train():

    device = get_device()

    class_list = list(TorchSigNarrowband._idx_to_name_dict.values())
    num_classes = len(class_list)
    batch_size = 16

    # Specify Transforms

    byol_transform = BYOLTransform(tensor_transform=Compose(
        [
            Normalize(norm=np.inf),
            ComplexTo2D(),
            ToTensor()
        ]
    ))
    target_transform = DescToClassIndex(class_list=class_list)

    datamodule = NarrowbandDataModule(
        root='./datasets/narrowband_test_QA',
        qa=True,
        impaired=False,
        transform=byol_transform,
        target_transform=target_transform,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # number of features for the embedding
    n_features = 2048

    backbone = ResNet1d(
        input_channels=2, n_features=n_features, resnet_version="50")
    model = BYOL(backbone=backbone,
                 num_ftrs=n_features,
                 hidden_dim=4096,
                 out_dim=256,
                 num_classes=num_classes,
                 batch_size_per_device=batch_size,
                 use_online_linear_eval=True)

    # Configure TensorBoardLogger
    training_path = "./train/byol"
    logger = TensorBoardLogger(training_path)

    # Configure ModelCheckpoint
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}",
        filename=f"byol-{type(model.backbone).__name__}-{{epoch:02d}}-{{train_loss:.2f}}",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
    )

    trainer = Trainer(max_epochs=100,
                      devices=1,
                      num_sanity_val_steps=1,
                      accelerator=device.type,
                      callbacks=[checkpoint_callback],
                      logger=logger)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    train()
