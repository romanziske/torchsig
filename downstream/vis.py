import cv2
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import detection_utils


def visualize_coco_dataset():
    # Register dataset
    register_coco_instances(
        "wideband_train",
        {},
        "datasets/wideband/coco/annotations/instances_train.json",
        "datasets/wideband/coco/train"
    )

    # Get metadata
    metadata = MetadataCatalog.get("wideband_train")
    dataset_dicts = DatasetCatalog.get("wideband_train")

    # Visualize 3 random samples
    for d in random.sample(dataset_dicts, 3):
        img = detection_utils.read_image(d["file_name"], format="L")

        print(f"Image shape: {img.shape}")
        visualizer = Visualizer(img, metadata=metadata, scale=1.0)
        vis = visualizer.draw_dataset_dict(d)

        # Show image
        cv2.imshow("COCO Visualization", vis.get_image())
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_coco_dataset()
