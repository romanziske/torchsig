import copy
import os
from pathlib import Path
from minio import Minio
import torch

from detectron2.engine import DefaultTrainer, launch
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

    client = Minio(
        os.environ['MINIO_URL'],
        access_key=os.environ['MINIO_ACCESS_KEY'],
        secret_key=os.environ['MINIO_SECRET_ACCESS_KEY'],
        cert_check=True
    )
 
    minio_prefix = Path("datasets/torchsig_wideband_2500_impaired")
    cwd_path = Path(__file__).resolve().parent
    datasets_path = os.path.join(cwd_path, "datasets", "wideband_torchsig")
 
    try:
 
        files = [
            "wideband_impaired_train/lock.mdb",
            "wideband_impaired_train/data.mdb",
            "wideband_impaired_val/lock.mdb",
            "wideband_impaired_val/data.mdb"
        ]
         
        for f in files:
             
            client.fget_object(
                bucket_name="iqdm-ai",
                object_name=os.path.join(minio_prefix, f).replace("\\", "/"),
                file_path=os.path.join(datasets_path, f),
            )
            
            print("Downloaded ", os.path.join(datasets_path, f))
            
    except Exception as e:
        print(e)

    # generate wideband dataset and convert it to COCO format
    converter = WidebandToSpectrogramCOCO(root_dir=datasets_path)

    converter.convert("train")
    converter.convert("val")

    coco_train_path = os.path.join(datasets_path, "coco/train")
    print(coco_train_path)
    coco_train_json_path = os.path.join(datasets_path, "coco/annotations/instances_train.json")
    print(coco_train_json_path)

    coco_val_path = os.path.join(datasets_path, "coco/val")
    print(coco_val_path)

    coco_val_json_path = os.path.join(datasets_path, "coco/annotations/instances_val.json")
    print(coco_val_json_path)
    
    # Register dataset

    register_coco_instances(
        "wideband_train",
        {},
        coco_train_json_path,
        coco_train_path
    )
    register_coco_instances(
        "wideband_val",
        {},
        coco_val_json_path,
        coco_val_path
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
    cfg.MODEL.DEVICE = "cuda"
    cfg.INPUT.FORMAT = "L"
    cfg.MODEL.PIXEL_MEAN = [128.0]  # Center data
    cfg.MODEL.PIXEL_STD = [128.0]   # Scale to ~[-1,1]

    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = 32
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.WARMUP_ITERS = 4000

    cfg.SOLVER.MAX_ITER = 10000
    #cfg.SOLVER.STEPS = ()
    #cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True

    # Save directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    train_detector()
