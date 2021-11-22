import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed)

    training_loop = instantiate(cfg.loop, cfg)

    train_cfg = cfg.training
    ckpt_path = os.path.join(os.getcwd(), 'checkpoints/')
    model_checkpoint = ModelCheckpoint(save_top_k=train_cfg.save_ckpts,
                                       monitor=train_cfg.monitor,
                                       mode=train_cfg.mode,
                                       verbose=True,
                                       filename='{epoch:03d}-{' + train_cfg.monitor + ':.3f}',
                                       dirpath=ckpt_path)

    early_stopping = EarlyStopping(monitor=train_cfg.monitor,
                                   patience=train_cfg.patience,
                                   mode=train_cfg.mode,
                                   verbose=True)

    if cfg.logger is not None:
        logger = instantiate(cfg.logger)
        if cfg.log_gradients:
            logger.experiment.watch(training_loop.model)
    else:
        # setting to True will use the default logger
        logger = True

    trainer = Trainer(max_epochs=train_cfg.epochs,
                      gpus=cfg.gpus,
                      logger=logger,
                      callbacks=[model_checkpoint, early_stopping],
                      accumulate_grad_batches=train_cfg.accumulate_batches)

    datamodule = instantiate(cfg.datamodule,
                             train_conf=cfg.training,
                             test_conf=cfg.testing,
                             num_workers=cfg.num_workers,
                             pin_memory=cfg.gpus > 0)

    trainer.fit(training_loop, datamodule=datamodule)


if __name__ == '__main__':
    train()
