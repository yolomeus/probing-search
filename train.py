import os
from pathlib import Path

import hydra
import wandb
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig):
    """Train a pytorch model specified by the config file, or continue from a checkpoint"""

    seed_everything(cfg.random_seed)

    if cfg.continue_from_checkpoint is not None:
        continue_training(cfg.continue_from_checkpoint)

    else:
        new_training(cfg)

    wandb.finish()


def new_training(cfg: DictConfig):
    """Start a new training run with the given configuration.

    :param cfg: the training configuration
    """
    trainer, datamodule, training_loop = instantiate_training(cfg)
    trainer.fit(training_loop, datamodule=datamodule)


def continue_training(ckpt_path):
    """Continue training, using state from a checkpoint. Note: This assumes the host machine to have the same
    capabilities as the one where the checkpoint was created, e.g. if gpu training was enabled, it will be enabled
    again.

    :param ckpt_path: path to the pytorch lightning checkpoint file.
    """

    ckpt_path = to_absolute_path(ckpt_path)
    # we need the old configuration
    run_dir = Path(ckpt_path).parent.parent
    run_conf_path = os.path.join(run_dir, '.hydra/config.yaml')
    ckpt_conf = OmegaConf.load(run_conf_path)

    run_id_file = os.path.join(run_dir, 'run_id.txt')
    with open(run_id_file, 'r') as fp:
        run_id = fp.read()

    trainer, datamodule, loop = instantiate_training(ckpt_conf, id=run_id, resume='must')
    trainer.fit(loop, datamodule=datamodule, ckpt_path=ckpt_path)


def instantiate_training(cfg: DictConfig, **logger_kwargs):
    """Instantiate trainer, datamodule and the training loop from a training configuration.

    :param logger_kwargs: any additional arguments to be passed to the logger's constructor.
    :param cfg: the training configuration
    :return: trainer, datamodule, training_loop
    """

    # create model
    model = instantiate(cfg.model)
    training_loop = instantiate(cfg.loop,
                                cfg,
                                model=model,
                                # params argument for optimizer constructor
                                optimizer={"params": model.parameters()})

    # create callbacks
    ckpt_path = os.path.join(os.getcwd(), 'checkpoints/')
    train_cfg = cfg.training
    model_checkpoint = ModelCheckpoint(save_top_k=train_cfg.save_ckpts,
                                       save_last=True,
                                       monitor=train_cfg.monitor,
                                       mode=train_cfg.mode,
                                       verbose=True,
                                       filename='{epoch:03d}-{' + train_cfg.monitor.replace('/', '_') + ':.3f}',
                                       dirpath=ckpt_path)

    early_stopping = EarlyStopping(monitor=train_cfg.monitor,
                                   patience=train_cfg.patience,
                                   mode=train_cfg.mode,
                                   verbose=True)

    # create logger
    if cfg.logger is not None:
        logger = instantiate(cfg.logger, tags=list(cfg.tags), **logger_kwargs)
        if cfg.log_gradients:
            logger.experiment.watch(training_loop.model)
    else:
        # setting to True will use the default logger
        logger = True

    # create trainer and datamodule
    trainer = Trainer(max_epochs=train_cfg.epochs,
                      gpus=cfg.gpus,
                      logger=logger,
                      callbacks=[model_checkpoint, early_stopping],
                      accumulate_grad_batches=train_cfg.accumulate_batches,
                      gradient_clip_val=train_cfg.gradient_clip_val,
                      gradient_clip_algorithm='norm')

    datamodule = instantiate(cfg.datamodule,
                             train_conf=cfg.training,
                             test_conf=cfg.testing,
                             num_workers=cfg.num_workers,
                             pin_memory=cfg.gpus > 0)

    return trainer, datamodule, training_loop


if __name__ == '__main__':
    train()
