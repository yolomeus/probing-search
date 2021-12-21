"""Abstraction for different training and testing procedures.
"""
from abc import abstractmethod, ABC
from os import path, getcwd

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class BaseTraining(ABC):
    """Base class for setting up and running training procedures.
    """

    def __init__(self, cfg: DictConfig):
        """
        :param cfg: the full training configuration.
        """
        self.cfg = cfg

        self.datamodule = self.build_datamodule()
        self.loop = self.build_loop()

        logger = self.build_logger(self.loop)
        callbacks = self.build_callbacks()
        self.trainer = self.build_trainer(logger, callbacks)

    @abstractmethod
    def run(self):
        """Override to specify the training procedure.
        """

    @abstractmethod
    def build_trainer(self, logger, callbacks):
        pass

    @abstractmethod
    def build_loop(self):
        pass

    @abstractmethod
    def build_datamodule(self):
        pass

    @abstractmethod
    def build_callbacks(self):
        pass

    @abstractmethod
    def build_logger(self, model):
        pass


class DefaultTraining(BaseTraining):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def run(self):
        self.trainer.fit(self.loop, datamodule=self.datamodule)
        self.trainer.test(datamodule=self.datamodule, ckpt_path='best')

    def build_trainer(self, logger, callbacks):
        train_cfg = self.cfg.training
        trainer = Trainer(
            max_epochs=train_cfg.epochs,
            gpus=self.cfg.gpus,
            logger=logger,
            callbacks=callbacks,
            accumulate_grad_batches=train_cfg.accumulate_batches,
            gradient_clip_val=train_cfg.gradient_clip_val,
            gradient_clip_algorithm='norm'
        )

        return trainer

    def build_loop(self):
        model = instantiate(self.cfg.model)
        training_loop = instantiate(
            self.cfg.loop,
            self.cfg,
            model=model,
            # params argument for optimizer constructor
            optimizer={'params': model.parameters()}
        )

        return training_loop

    def build_datamodule(self):
        return instantiate(
            self.cfg.datamodule,
            train_conf=self.cfg.training,
            test_conf=self.cfg.testing,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.gpus > 0
        )

    def build_callbacks(self):
        train_cfg = self.cfg.training
        ckpt_path = path.join(getcwd(), 'checkpoints/')

        model_checkpoint = ModelCheckpoint(
            save_top_k=train_cfg.save_ckpts,
            save_last=True,
            monitor=train_cfg.monitor,
            mode=train_cfg.mode,
            verbose=True,
            filename='{epoch:03d}-{' + train_cfg.monitor.replace('/', '_') + ':.3f}',
            dirpath=ckpt_path
        )

        early_stopping = EarlyStopping(
            monitor=train_cfg.monitor,
            patience=train_cfg.patience,
            mode=train_cfg.mode,
            verbose=True
        )

        return [model_checkpoint, early_stopping]

    def build_logger(self, model):
        if self.cfg.logger is not None:
            logger = instantiate(self.cfg.logger, tags=list(self.cfg.tags))
            if self.cfg.log_gradients:
                logger.experiment.watch(model)
            return logger

        # setting to True will use the default logger
        return True
