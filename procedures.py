"""Abstraction for different training and testing procedures.
"""
import logging
import os.path
from abc import abstractmethod, ABC
from os import path, getcwd

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule.probing_datamodule import MDLProbingDataModule
from metrics import MDL, Compression


class Procedure(ABC):
    """Base class for setting up and running training procedures.
    """

    @abstractmethod
    def run(self):
        """Override to specify the training procedure.
        """


class BaseTraining(Procedure, ABC):
    """Defines methods for setting up parts of a basic training procedure.
    """

    def __init__(self, cfg: DictConfig):
        self.log = logging.getLogger('.'.join([self.__module__, self.__class__.__name__]))
        self.cfg = cfg

    def build_trainer(self, logger, callbacks=None, **kwargs):
        train_cfg = self.cfg.training
        trainer = Trainer(
            max_epochs=train_cfg.epochs,
            gpus=self.cfg.gpus,
            logger=logger,
            callbacks=callbacks,
            accumulate_grad_batches=train_cfg.accumulate_batches,
            gradient_clip_val=train_cfg.gradient_clip_val,
            gradient_clip_algorithm='norm',
            **kwargs
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

    def build_callbacks(self, ckpt_path=None):
        train_cfg = self.cfg.training
        ckpt_path = path.join(getcwd(), 'checkpoints/') if ckpt_path is None else ckpt_path

        model_checkpoint = ModelCheckpoint(
            save_top_k=train_cfg.save_ckpts,
            monitor=train_cfg.monitor,
            mode=train_cfg.mode,
            verbose=True,
            filename='epoch-{epoch:03d}-' + train_cfg.monitor.replace('/', '_') + '-{' + train_cfg.monitor + ':.3f}',
            auto_insert_metric_name=False,
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
            logger = instantiate(self.cfg.logger, **kwargs)
            if self.cfg.log_gradients:
                logger.experiment.watch(model)
            return logger

        # setting to True will use the default logger
        return True


class DefaultTraining(BaseTraining):
    """A standard training run with a single fit and test run.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.datamodule = self.build_datamodule()
        self.loop = self.build_loop()

        self.logger = self.build_logger(self.loop)
        self.trainer = self.build_trainer(self.logger, self.build_callbacks())

    def run(self):
        self.trainer.fit(self.loop, datamodule=self.datamodule)
        self.trainer.test(datamodule=self.datamodule, ckpt_path='best')


class MDLOnlineCoding(BaseTraining):
    """Computes Minimum Description Length of a probing model using online-coding, i.e. training on increasing portions
    of the dataset.
    """

    def __init__(self, datamodule: MDLProbingDataModule, experiment, limit_mdl_val_steps: int, cfg: DictConfig):
        super().__init__(cfg)

        self.experiment = experiment
        self.logger = self.build_logger(self.build_loop(), experiment=self.experiment)

        num_classes = cfg.datamodule.dataset.num_classes
        self.limit_mdl_val_steps = limit_mdl_val_steps
        self.mdl = MDL(len(datamodule.portions) - 1, num_classes)
        self.compression = Compression(num_classes)

        # we need to initialize the train ids first if we want to access portion sizes etc. before training
        self.datamodule = datamodule
        self.datamodule.prepare_data()
        self.datamodule.setup()

    def run(self):
        self.log.info('Performing multi-portion training')
        self._run_online_coding()
        mdl, compression = self._compute_metrics()
        self.logger.log_metrics({'mdl': mdl, 'compression': compression})

    def _run_online_coding(self):
        """Run the online-coding procedure to compute mdl and compression. For this we train on multiple portions of the
        full training set and evaluate each individually trained model.
        """
        num_portions = len(self.datamodule.portions)
        datamodule = self.datamodule
        for i in range(num_portions - 1):
            portion_percentage = datamodule.current_portion_percentage
            portion_size = datamodule.current_portion_size

            self.log.info(f'Training on {portion_percentage} ({portion_size} instances) of training set')

            # on each portion, we train the model from scratch
            loop = self.build_loop()

            # logger with custom postfix for current portion
            portion_log = self._postfix_logger(loop, i)
            portion_log.log_metrics({'portion_percentage': portion_percentage,
                                     'portion_size': portion_size})

            ckpt_path = os.path.join(os.getcwd(), f'checkpoints/mdl/portion_{i:02d}/')
            trainer = self.build_trainer(logger=portion_log,
                                         callbacks=self.build_callbacks(ckpt_path),
                                         limit_val_batches=self.limit_mdl_val_steps)
            trainer.fit(loop, datamodule)

            with torch.no_grad():
                self._update_mdl(trainer, loop, i)

            self.datamodule.next_portion()

            # cleanup prevents a multiprocessing issue where already closed handles are closed again
            del loop
            del trainer

    def _update_mdl(self, trainer, loop, i):
        # for MDL: using the model trained on the set portion(i),
        # predict on the set difference (portion(i + 1) - portion(i)) (See MDLProbingDataModule.pred_ds)
        preds = trainer.predict(loop, self.datamodule, ckpt_path='best')

        for y_pred_batch, y_batch in preds:
            # update mdl metric for the i-th portion
            self.mdl.update(
                y_pred_batch,
                y_batch,
                portion_idx=i,
                first_portion_size=None if i != 0 else self.datamodule.num_targets_portion
            )

    def _compute_metrics(self):
        """Compute final value of mdl and compression from aggregated states.
        :return: tuple: mdl, compression
        """
        mdl = self.mdl.compute()
        compression = self.compression(mdl, self.datamodule.num_targets_total)

        self.mdl.reset()
        self.compression.reset()

        return mdl, compression

    def _postfix_logger(self, loop, i):
        return self.build_logger(loop,
                                 postfix=f'/portion_{i:02d}',
                                 experiment=self.experiment)


class MDLProbeTraining(Procedure):
    """First, compute and log MDL and Compression using online-coding, then perform a full training run.
    """

    def __init__(self, cfg: DictConfig, limit_mdl_val_steps: int):
        self.default_training = DefaultTraining(cfg)
        self.online_coding = MDLOnlineCoding(self.default_training.datamodule,
                                             self.default_training.logger.experiment,
                                             limit_mdl_val_steps,
                                             cfg)

    def run(self):
        self.online_coding.run()
        del self.online_coding
        self.default_training.run()
