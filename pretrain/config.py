from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TrainingConfig:
    # Dataset config
    dataset: Literal['narrowband', 'wideband'] = 'narrowband'
    root: str = './datasets/narrowband'
    batch_size: int = 16
    num_workers: int = 0
    qa: bool = True
    impaired: bool = False
    family: bool = False

    # Model config
    n_features: int = 2048
    hidden_dim: int = 4096
    out_dim: int = 256
    ssl_model: Literal['byol'] = 'byol'
    backbone: Literal['resnet50'] = 'resnet50'
    online_linear_eval: bool = False

    # Training config
    training_path: str = './train'
    num_epochs: int = 100
    spectrogram: bool = False
    nfft: int = 512

    # Export config
    checkpoint: str = 'best'


@dataclass
class EvaluationConfig:
    # model config
    model_path: str
    backbone: Literal['resnet50'] = 'resnet50'
    spectrogram: bool = False
    n_features: int = 2048

    # dataset config
    dataset: Literal['narrowband', 'wideband'] = 'narrowband'
    root: str = './datasets/narrowband'
    batch_size: int = 16
    num_workers: int = 0
    qa: bool = True
    impaired: bool = False
    family: bool = False
    nfft: int = 512

    # Export config
    tsne: bool = True
    knn: bool = True
    n_neighbors: int = 5
    export_path: str = '.'
