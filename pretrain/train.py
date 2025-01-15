import os
from lightning.pytorch import Trainer, callbacks
from lightning.pytorch.loggers import TensorBoardLogger

from config import TrainingConfig
from callbacks import ModelAndBackboneCheckpoint
from utils import get_dataset, get_device, get_ssl_model, parse_args, print_config


def train(config: TrainingConfig):

    device = get_device()
    datamodule = get_dataset(config)

    if not config.online_linear_eval:
        datamodule.val_dataloader = None

    ssl_model = get_ssl_model(config)

    # Configure TensorBoardLogger
    logger = TensorBoardLogger(
        os.path.join(config.training_path, config.ssl_model)
    )

    # Configure ModelCheckpoint
    checkpoint_callback = ModelAndBackboneCheckpoint(
        dirpath=f"{logger.save_dir}/lightning_logs/version_{logger.version}",
        filename=(
            f"{config.ssl_model}"
            f"-{config.backbone}"
            f"-{config.dataset}"
            f"-{'spec' if config.spectrogram else 'iq'}"
            f"-e{{epoch:d}}"
            f"-b{config.batch_size}"
            f"-loss{{train_loss:.3f}}"
        ),
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=config.num_epochs,
        devices=1,
        accelerator=device.type,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model=ssl_model, datamodule=datamodule)


if __name__ == '__main__':
    config = parse_args("TrainingConfig")
    print_config(config)
    train(config)
