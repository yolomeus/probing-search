import os

import hydra
from hydra.utils import get_class, to_absolute_path, instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer, LightningDataModule

from logger.loggers import WandbMinMaxLogger


def test_checkpoint(ckpt_path: str, test_cfg: DictConfig, trainer: Trainer, datamodule: LightningDataModule):
    """Load model state from checkpoint and test it.

    :param ckpt_path: path to the the checkpoint file.
    :param test_cfg: test configuration.
    :param trainer: trainer used for testing.
    :param datamodule: datamodule to test on.
    """

    model_cls = get_class(test_cfg.loop._target_)
    model = model_cls.load_from_checkpoint(ckpt_path)
    # make sure we're using the current test config and not the saved one
    model.test_conf = test_cfg.testing
    trainer.test(model, datamodule=datamodule)


@hydra.main(config_path='conf', config_name='config')
def test(cfg: DictConfig):
    """Test a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed)

    datamodule = instantiate(cfg.datamodule,
                             train_conf=cfg.training,
                             test_conf=cfg.testing,
                             num_workers=cfg.num_workers,
                             pin_memory=cfg.gpus > 0)

    logger = WandbMinMaxLogger(tags=['test']) if cfg.wandb_log else True

    log_dir = to_absolute_path(cfg.testing.log_dir)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')

    top_k = os.listdir(ckpt_dir)[-cfg.testing.test_best_k:]
    for file in top_k:
        ckpt_path = os.path.join(ckpt_dir, file)
        trainer = Trainer(gpus=cfg.gpus, logger=logger, resume_from_checkpoint=ckpt_path)
        test_checkpoint(ckpt_path, cfg, trainer, datamodule)


if __name__ == '__main__':
    test()
