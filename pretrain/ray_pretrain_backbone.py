
import ray
from ray.train.torch import TorchTrainer
from ray.air import RunConfig, ScalingConfig

from config import TrainingConfig
from pretrain_backbone import train, parse_args, print_config


def train_on_ray(config: TrainingConfig):
    # Initialize Ray
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker=train,
        train_loop_config=config,
        run_config=RunConfig(
            name="ssl_pretraining",
        ),
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=False,
        )
    )

    results = trainer.fit()
    print(f"Training completed: {results}")


if __name__ == "__main__":
    config = parse_args("TrainingConfig")
    print_config(config)
    train_on_ray(config)
