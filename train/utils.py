import argparse
from dataclasses import MISSING, Field, fields
from typing import Any, Literal, Union

import numpy as np
import torch

from torchsig.datasets.datamodules import NarrowbandDataModule, TorchSigDataModule, WidebandDataModule
from torchsig.datasets.signal_classes import torchsig_signals
from torchsig.models.iq_models.resnet.resnet1d import ResNet1d
from torchsig.models.spectrogram_models.resnet.resnet import resnet
from torchsig.models.ssl_models.byol import BYOL
from torchsig.transforms.byol_transform import BYOLTransform
from torchsig.transforms.target_transforms import DescToClassIndex, DescToFamilyIndex
from torchsig.transforms.transforms import ComplexTo2D, Compose, Identity, Lambda, Normalize, Spectrogram, ToSpectrogramTensor, ToTensor, Transform

from config import EvaluationConfig, TrainingConfig


def collate_fn(batch: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Custom collate for 2 views"""

    views, targets = zip(*batch)

    view1s, view2s = zip(*views)

    return (
        torch.stack(view1s),
        torch.stack(view2s)
    ), torch.tensor(targets)


def collate_fn_evaluation(batch):
    # Extract tensors and targets
    tensors, targets = zip(*batch)

    # Stack tensors into single batch
    tensors = torch.stack(tensors)

    # Convert targets to tensor
    targets = torch.tensor(targets)

    return tensors, targets


def get_collate_fn(config: Union[TrainingConfig, EvaluationConfig]):

    if isinstance(config, EvaluationConfig):
        return collate_fn_evaluation
    else:
        return collate_fn


def get_device() -> torch.device:
    """Get the appropriate PyTorch device for the current system.

    This function checks available hardware acceleration in the following order:
    1. Apple Silicon MPS (Metal Performance Shaders) for Mac
    2. CUDA for NVIDIA GPUs
    3. CPU as fallback

    Returns:
        torch.device: Device object representing the best available compute device:
            - 'mps' for Apple Silicon Macs
            - 'cuda' for systems with NVIDIA GPUs
            - 'cpu' for systems without GPU acceleration

    Example:
        >>> device = get_device()
        >>> model = Model().to(device)
    """

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_class_list(config: Union[TrainingConfig, EvaluationConfig]) -> list:
    if config.family:
        # Get unique sorted family names from class-family dict
        return sorted(list(set(torchsig_signals.family_dict.values())))
    return torchsig_signals.class_list


def get_tensor_transform(config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
    if config.spectrogram:
        return Compose(
            [
                Spectrogram(
                    nperseg=512,
                    noverlap=0,
                    nfft=512,
                    mode='psd',
                ),
                Normalize(norm=np.inf, flatten=True),
                ToSpectrogramTensor(),
            ]
        )
    else:
        return Compose(
            [
                Normalize(norm=np.inf),
                ComplexTo2D(),
                ToTensor(),
            ]
        )


def get_transform(config: Union[TrainingConfig, EvaluationConfig]) -> Transform:
    tensor_transform = get_tensor_transform(config)

    if isinstance(config, EvaluationConfig):
        return tensor_transform

    if config.ssl_model == 'byol':
        return BYOLTransform(tensor_transform=tensor_transform)


def get_target_transform(config: Union[TrainingConfig, EvaluationConfig]):
    if config.dataset == 'narrowband':
        if config.family:
            return DescToFamilyIndex()
        return DescToClassIndex(class_list=get_class_list(config))

    if config.dataset == 'wideband':
        # for wideband dataset we don't need target transform, because we can
        # not use it for online linear evaluation

        return Lambda(lambda x: 0)


def get_dataset(config: TrainingConfig) -> TorchSigDataModule:

    transform = get_transform(config)
    target_transform = get_target_transform(config)

    if config.dataset == 'narrowband':

        return NarrowbandDataModule(
            root=config.root,
            qa=config.qa,
            impaired=config.impaired,
            transform=transform,
            target_transform=target_transform,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=get_collate_fn(config),
        )

    if config.dataset == 'wideband':

        return WidebandDataModule(
            root=config.root,
            qa=config.qa,
            impaired=config.impaired,
            transform=transform,
            target_transform=target_transform,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            num_classes=len(get_class_list(config)),
            collate_fn=get_collate_fn(config),
        )


def get_backbone(config: Union[TrainingConfig, EvaluationConfig]) -> torch.nn.Module:
    if config.backbone == 'resnet50':
        if config.spectrogram:
            return resnet(
                input_channels=1,
                n_features=config.n_features,
                resnet_version="50"
            )
        else:
            return ResNet1d(
                input_channels=2,
                n_features=config.n_features,
                resnet_version="50",
            )


def get_ssl_model(config: TrainingConfig) -> torch.nn.Module:

    backbone = get_backbone(config)

    if config.ssl_model == 'byol':
        return BYOL(
            backbone=backbone,
            num_ftrs=config.n_features,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            batch_size_per_device=config.batch_size,
            use_online_linear_eval=config.online_linear_eval,
            num_classes=len(get_class_list(config)),
        )


def get_config_fields(mode: Literal["TrainingConfig", "EvaluationConfig"]) -> tuple[Field[Any], ...]:
    """Get all unique fields from both config types"""
    if mode == "EvaluationConfig":
        return fields(EvaluationConfig)
    else:
        return fields(TrainingConfig)


def get_arg_type(field_type: Any) -> type:
    """Get appropriate argument type for field"""
    type_str = str(field_type)

    # Type mapping
    type_map = {
        "<class 'str'>": str,
        "<class 'int'>": int,
        "<class 'float'>": float,
        "<class 'bool'>": bool
    }

    if type_str in type_map:
        return type_map[type_str]
    return str


def parse_args(mode: Literal["TrainingConfig", "EvaluationConfig"]) -> Union[TrainingConfig, EvaluationConfig]:
    parser = argparse.ArgumentParser(
        description='Train SSL models on RF signals')

    for field in get_config_fields(mode):
        field_type = field.type
        field_default = field.default
        field_help = field.metadata.get('help', f'{field.name} parameter')

        kwargs: dict[str, Any] = {'help': field_help}

        if field_default is MISSING:
            kwargs['required'] = True
            kwargs['type'] = get_arg_type(field_type)
        else:
            if isinstance(field_default, bool):
                kwargs['action'] = 'store_true' if not field_default else 'store_false'
            else:
                kwargs['type'] = type(field_default)
                kwargs['default'] = field_default

        if str(field_type).startswith('typing.Literal'):
            choices = field_type.__args__
            kwargs['choices'] = choices

        parser.add_argument(f'--{field.name}', **kwargs)

    args = parser.parse_args()
    config_class = EvaluationConfig if mode == "EvaluationConfig" else TrainingConfig
    return config_class(**vars(args))


def print_config(config: Union[TrainingConfig, EvaluationConfig]) -> None:
    """Print config in a structured format"""
    # ANSI color codes
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'

    print(f"\n{BLUE}Configuration:{ENDC}")

    # Get fields directly from config instance
    for field in fields(config):
        value = getattr(config, field.name)
        print(f"  {CYAN}{field.name}:{ENDC} {GREEN}{value}{ENDC}")
