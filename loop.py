"""Training/testing loops specified by pytorch-lightning models. Unlike in standard pytorch-lightning, the loop should
encapsulate the model instead of being bound to it by inheritance. This way, the same model can be trained with
multiple different procedures, without having to duplicate model code by subclassing.
"""
from abc import ABC

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.nn import Module, ModuleDict, Sigmoid, Softmax, Identity
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.retrieval import RetrievalMetric

from datamodule import DatasetSplit
from metrics import TrecNDCG


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

    def __init__(self, hparams: DictConfig, model: Module, optimizer: Optimizer, loss: Module, train_metrics,
                 val_metrics, test_metrics, to_probabilities):
        """
        :param hparams: contains all hyperparameters.
        """
        super().__init__(hparams)

        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.train_metrics = ModuleDict(train_metrics)
        self.val_metrics = ModuleDict(val_metrics)
        self.test_metrics = ModuleDict(test_metrics)

        self.to_probabilities = self._probs_module(to_probabilities)

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
        loss = self.log_metrics(y_pred, y_true, DatasetSplit.TRAIN)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        self.log_metrics(y_pred, y_true, DatasetSplit.VALIDATION)

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        self.log_metrics(y_pred, y_true, DatasetSplit.TEST)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y_true = batch
        y_pred = self.model(x)
        return y_pred, y_true

    def log_metrics(self, y_pred, y_true, split: DatasetSplit, **kwargs):
        y_prob = self.to_probabilities(y_pred)

        metrics = self._select_metrics(split)
        for name, metric in metrics.items():
            self.call_metric(y_prob, y_true, metric, **kwargs)
            self.log(f'{split.value}/{name}',
                     metric,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(y_true))

        loss = self.loss(y_pred, y_true)
        self.log(f'{split.value}/loss', loss, on_step=False, on_epoch=True, batch_size=len(y_true))

        if split == DatasetSplit.TRAIN:
            return loss

    def _select_metrics(self, split):
        if split == DatasetSplit.TRAIN:
            return self.train_metrics
        elif split == DatasetSplit.TEST:
            return self.test_metrics

        return self.val_metrics

    def _probs_module(self, name: str):
        if name.lower() == 'sigmoid':
            return Sigmoid()
        elif name.lower() == 'softmax':
            return Softmax(dim=-1)
        elif name.lower() in [None, 'identity']:
            return Identity()
        else:
            raise NotImplementedError

    def call_metric(self, y_prob, y_true, metric, **kwargs):
        """overwrite if metrics require special arguments etc."""
        metric(y_prob, y_true)


class RankingLoop(DefaultClassificationLoop):

    def training_step(self, batch, batch_idx):
        _, _, x, y_true, _ = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y_true)

        self.log('train/loss', loss.item(), on_step=True, on_epoch=True, batch_size=len(y_true))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        q_ids, _, x, y_true, y_rank = batch
        y_pred = self.model(x)
        self.log_metrics(y_pred, y_true, DatasetSplit.VALIDATION, indexes=q_ids, y_rank=y_rank)

    def test_step(self, batch, batch_idx):
        q_ids, doc_ids, x, y_true, y_rank = batch
        y_pred = self.model(x)
        self.log_metrics(y_pred, y_true, DatasetSplit.TEST, indexes=q_ids, y_rank=y_rank)

        return {'q_ids': q_ids, 'doc_ids': doc_ids, 'y_pred': y_pred}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        q_ids, _, x, y_true, _ = batch
        y_pred = self.model(x)
        return y_pred, y_true

    def call_metric(self, y_prob, y_true, metric, **kwargs):
        if issubclass(type(metric), TrecNDCG):
            # for ndcg we use integer scores instead of binary labels
            metric(y_prob, kwargs['y_rank'], indexes=kwargs['indexes'])
        elif issubclass(type(metric), RetrievalMetric):
            metric(y_prob, y_true, indexes=kwargs['indexes'])
        else:
            metric(y_prob, y_true)
