"""Abstraction for different training and testing procedures.
"""
import logging
from abc import abstractmethod, ABC
from os import path, getcwd

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule.probing_datamodule import MDLProbingDataModule
from metrics import MDL, Compression


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

        self.logger = self.build_logger(self.loop)
        callbacks = self.build_callbacks()
        self.trainer = self.build_trainer(self.logger, callbacks)

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
    """Default training setup for a single dataset training run.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

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

    def build_logger(self, model, **kwargs):
        if self.cfg.logger is not None:
            logger = instantiate(self.cfg.logger, tags=list(self.cfg.tags), **kwargs)
            if self.cfg.log_gradients:
                logger.experiment.watch(model)
            return logger

        # setting to True will use the default logger
        return True


class MDLProbeTraining(DefaultTraining):
    """Compute MDL using online-coding, before performing a full training run.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.log = logging.getLogger(self.__module__)

        assert isinstance(self.datamodule, MDLProbingDataModule), 'MDLProbeTraining expects an MDL specific Datamodule.'

        self.experiment = self.logger.experiment

        num_classes = cfg.datamodule.dataset.num_classes
        self.mdl = MDL(len(self.datamodule.portions) - 1, num_classes)
        self.compression = Compression(num_classes)

    def run(self):
        # online coding run
        self.log.info('Performing multi-portion training')
        self._run_online_coding()
        self._log_mdl()

        assert self.datamodule.current_portion_percentage == 1.0, 'The final run should be over the full dataset'

        # normal run
        self.log.info('Starting full run')
        self.trainer = self.build_trainer(logger=self.logger, callbacks=self.build_callbacks())
        super().run()

    def _run_online_coding(self):
        """Run the online-coding procedure to compute mdl and compression. For this we train on multiple portions of the
        full training set and evaluate each individually trained model.
        """
        num_portions = len(self.datamodule.portions)
        for i in range(num_portions - 1):
            portion_percentage = self.datamodule.current_portion_percentage
            portion_size = self.datamodule.current_portion_size
            self.log.info(
                f'Training on {portion_percentage} '
                f'({portion_size} instances) of training set'
            )

            # on each portion, we train the model from scratch
            self.loop = self.build_loop()

            # logger with custom postfix for current portion
            portion_logger = self.build_logger(self.loop, postfix=f'/portion_{i:02d}', experiment=self.experiment)
            portion_logger.log_metrics({'portion_percentage': portion_percentage, 'portion_size': portion_size})

            self.trainer = self.build_trainer(logger=portion_logger, callbacks=self.build_callbacks())
            self.trainer.fit(self.loop, self.datamodule)

            self._update_mdl(i)
            self.datamodule.next_portion()

    def _update_mdl(self, i):
        # for MDL: using the model trained on the set portion(i),
        # predict on the set difference (portion(i + 1) - portion(i)) (See MDLProbingDataModule.pred_ds)
        preds = self.trainer.predict(self.loop, self.datamodule, ckpt_path='best')
        y_pred, y_true = self._unpack_predictions(preds)

        # update mdl metric for the i-th portion
        self.mdl.update(
            y_pred,
            y_true,
            portion_idx=i,
            first_portion_size=None if i != 0 else self.datamodule.num_targets_portion
        )

    def _log_mdl(self):
        mdl = self.mdl.compute()
        compression = self.compression(mdl, self.datamodule.num_targets_total)
        self.logger.log_metrics({'mdl': mdl, 'compression': compression})

    @staticmethod
    def _unpack_predictions(preds):
        y_pred, y_true = zip(*preds)
        y_pred, y_true = map(torch.cat, [y_pred, y_true])
        return y_pred, y_true
