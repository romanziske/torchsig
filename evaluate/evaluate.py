import os
import numpy as np
import torch
from torch.serialization import add_safe_globals
from tqdm import tqdm
from timm.models.resnet import ResNet

from knn import EvaluateKNN
from torchsig.datasets.signal_classes import torchsig_signals
from torchsig.models.ssl_models.byol import BYOL
from tsne import VisualizeTSNE


from torchsig.datasets.datamodules import NarrowbandDataModule
from torchsig.datasets.torchsig_narrowband import TorchSigNarrowband
from torchsig.models.iq_models.resnet.resnet1d import ResNet1d
from torchsig.transforms.target_transforms import DescToClassIndex, DescToFamilyIndex, DescToFamilyName
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
    # Extract tensors and targets
    tensors, targets = zip(*batch)

    # Stack tensors into single batch
    tensors = torch.stack(tensors)

    # Convert targets to tensor
    targets = torch.tensor(targets)

    return tensors, targets


def get_class_list(family: bool = False) -> list:
    if family:
        # Get unique sorted family names from class-family dict
        return sorted(list(set(torchsig_signals.family_dict.values())))
    return torchsig_signals.class_list


def convert_idx_to_name(idx: int, family: bool = False) -> str:
    """Convert index to either class or family name"""
    return get_class_list(family)[idx]


def evaluate():
    batch_size = 16

    device = get_device()
    family = True
    class_list = get_class_list(family=family)

    # Specify Transforms

    transform = Compose(
        [
            Normalize(norm=np.inf),
            ComplexTo2D(),
            ToTensor()
        ]
    )
    target_transform = Compose(
        [
            DescToFamilyIndex(),
        ]
    )

    datamodule = NarrowbandDataModule(
        root='./datasets/narrowband_test_QA',
        qa=True,
        impaired=False,
        transform=transform,
        target_transform=target_transform,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = ResNet1d(
        input_channels=2,
        n_features=2048,
        resnet_version="50"
    )

    checkpoint = torch.load(
        "backbone.pth",
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    representations = []
    labels = []

    datamodule.setup("fit")
    val_dataloader = datamodule.train_dataloader()
    with torch.no_grad():  # No gradient needed

        for x, targets in tqdm(val_dataloader):
            x = x.to(device)

            z = model(x)  # Run inference
            # Move features back to CPU and convert to numpy
            representations.extend(z.cpu().numpy())

            # Use .item() to get Python number from tensor
            labels.extend(
                [convert_idx_to_name(t.item(), family) for t in targets])

    representations = np.array(representations)
    labels = np.array(labels)
    print(
        f"Finished calculating representations (shape {representations.shape})")

    print("Start t-SNE visualization...")
    plot_path = f"tsne_plot_{"narrowband1"}.png"
    VisualizeTSNE(
        x=representations,
        y=labels,
        class_list=class_list,
    ).visualize(save_path=plot_path)
    print(f"t-SNE plot saved at {plot_path}")

    print("Start KNN evaluation...")
    accuracy = EvaluateKNN(representations, labels, n_neighbors=3).evaluate()
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate()
