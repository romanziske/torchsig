import numpy as np
import torch
from tqdm import tqdm

from knn import EvaluateKNN
from config import EvaluationConfig
from utils import get_backbone, get_class_list, get_dataset, get_device, parse_args, print_config
from tsne import VisualizeTSNE


def convert_idx_to_name(idx: int, config: EvaluationConfig) -> str:
    """Convert index to either class or family name"""
    return get_class_list(config)[idx]


def evaluate(config: EvaluationConfig):

    if not config.model_path:
        raise ValueError("model_path is required for evaluation")

    device = get_device()
    datamodule = get_dataset(config)

    model = get_backbone(config)
    checkpoint = torch.load(
        config.model_path,
        map_location=device,
        weights_only=True,
    )

    # Load model state dict
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
                [convert_idx_to_name(t.item(), config) for t in targets])

    representations = np.array(representations)
    labels = np.array(labels)
    print(
        f"Finished calculating representations (shape {representations.shape})")

    print("Start t-SNE visualization...")
    model_name = config.model_path.split("/")[-1].split(".")[0]
    plot_path = f"tsne_plot_{model_name}.png"
    VisualizeTSNE(
        x=representations,
        y=labels,
        class_list=get_class_list(config),
    ).visualize(save_path=plot_path)
    print(f"t-SNE plot saved at {plot_path}")

    print("Start KNN evaluation...")
    accuracy = EvaluateKNN(representations, labels, n_neighbors=3).evaluate()
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    config = parse_args("EvaluationConfig")
    print_config(config)
    evaluate(config)
