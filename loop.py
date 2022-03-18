"""Training/testing loops specified by pytorch-lightning models. Unlike in standard pytorch-lightning, the loop should
encapsulate the model instead of being bound to it by inheritance. This way, the same model can be trained with
multiple different procedures, without having to duplicate model code by subclassing.
"""
from abc import ABC
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datamodule import DatasetSplit
from logger.utils import Metrics, write_trec_eval_file


class AbstractBaseLoop(LightningModule, ABC):
    """Abstract base class for implementing a training loop for a pytorch model.
    """

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)


class DefaultClassificationLoop(AbstractBaseLoop):
    """Default wrapper for training/testing a pytorch module using pytorch-lightning. Assumes a standard classification
    task with instance-label pairs (x, y) and a loss function that has the signature loss(y_pred, y_true).
    """

    def __init__(self, hparams: DictConfig, model: Module, optimizer: Optimizer, loss: Module):
        """
        :param hparams: contains all hyperparameters.
        """
        super().__init__(hparams)

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = Metrics(self.loss, hparams.metrics.metrics_list, hparams.metrics.to_probabilities)

    def configure_optimizers(self):
        train_conf = self.hparams.training
        return {'optimizer': self.optimizer,
                'lr_scheduler': {'scheduler': ReduceLROnPlateau(self.optimizer,
                                                                factor=train_conf.schedule_factor,
                                                                patience=train_conf.schedule_patience),
                                 'monitor': train_conf.monitor}}

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.metrics.metric_log(self, y_pred, y_true, DatasetSplit.TRAIN)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        self.metrics.metric_log(self, y_pred, y_true, DatasetSplit.VALIDATION)

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        self.metrics.metric_log(self, y_pred, y_true, DatasetSplit.TEST)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y_true = batch
        y_pred = self.model(x)
        return y_pred, y_true


class RankingLoop(DefaultClassificationLoop):

    def training_step(self, batch, batch_idx):
        _, _, x, y_true = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y_true)

        self.log('train/loss', loss, on_step=True, on_epoch=True, batch_size=len(y_true))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        q_ids, _, x, y_true = batch
        y_pred = self.model(x)
        self.metrics.metric_log(self, y_pred, y_true, DatasetSplit.VALIDATION, indexes=q_ids)

    def test_step(self, batch, batch_idx):
        q_ids, doc_ids, x, y_true = batch
        y_pred = self.model(x)
        self.metrics.metric_log(self, y_pred, y_true, DatasetSplit.TEST, indexes=q_ids)

        return {'q_ids': q_ids, 'doc_ids': doc_ids, 'y_pred': y_pred}

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        # q_id to doc_id to score
        results_dict = defaultdict(dict)
        for output in outputs:
            for q_id, doc_id, pred in zip(output['q_ids'], output['doc_ids'], output['y_pred']):
                results_dict[q_id.item()][doc_id.item()] = pred[-1].item()

        write_trec_eval_file(Path('./trec_predictions.csv'), results_dict, self.model.subject_model.model_name)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        q_ids, x, y_true = batch
        y_pred = self.model(x)
        return y_pred, y_true
