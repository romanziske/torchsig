import copy
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils

from to_coco_dataset import WidebandToSpectrogramCOCO


def mapper(dataset_dict):
    # it will be modified by code below
    dataset_dict = copy.deepcopy(dataset_dict)

    # read image and store as torch tensor
    image = detection_utils.read_image(dataset_dict["file_name"], format="L")
    # convert to writable array
    image_shape = image.shape[:2]  # (h, w, c) -> (h, w)

    # transform the image to tensor (c, h, w)
    image_tensor = torch.tensor(image)
    dataset_dict["image"] = torch.as_tensor(image_tensor.permute(2, 0, 1))

    # annotations to detectron2 instances
    dataset_dict["instances"] = detection_utils.annotations_to_instances(
        dataset_dict["annotations"], image_size=image_shape)

    return dataset_dict


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=mapper
        )


def setup_datasets():

    # generate wideband dataset and convert it to COCO format
    converter = WidebandToSpectrogramCOCO("datasets/wideband_torchsig")

    converter.convert("train")
    converter.convert("val")

    # Register dataset
    register_coco_instances(
        "wideband_train",
        {},
        "datasets/wideband_torchsig/coco/annotations/instances_train.json",
        "datasets/wideband_torchsig/coco/train"
    )
    register_coco_instances(
        "wideband_val",
        {},
        "datasets/wideband_torchsig/coco/annotations/instances_val.json",
        "datasets/wideband_torchsig/coco/val"
    )


def train_detector():

    setup_datasets()

    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("wideband_train",)
    cfg.DATASETS.TEST = ("wideband_val",)
    cfg.MIN_SIZE_TRAIN = (256,)  # Keep fixed size

    # Model parameters
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 61  # Number of signal classes
    cfg.MODEL.DEVICE = "cpu"
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.PIXEL_MEAN = [128.0]  # Center data
    cfg.MODEL.PIXEL_STD = [128.0]   # Scale to ~[-1,1]

    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (7000, 9000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # Save directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    train_detector()
