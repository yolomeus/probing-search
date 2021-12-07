"""Training loop related utilities.
"""
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module, ModuleList, Softmax, Sigmoid

from datamodule import DatasetSplit


class Metrics(Module):
    """Stores and manages metrics during training/testing for log.
    """

    def __init__(self, loss: Module, metrics_configs: List[DictConfig], to_probabilities: str):
        """

        :param loss: the loss module for computing the loss metric.
        :param metrics_configs: dict configs for each metric to instantiate.
        :param to_probabilities: either 'sigmoid' or 'softmax', used to convert raw model predictions into
        probabilities before passing them into a metric function.
        """
        super().__init__()

        per_split_metrics = [[] if metrics_configs is None else [instantiate(metric) for metric in metrics_configs]
                             for _ in range(3)]
        self.train_metrics, self.val_metrics, self.test_metrics = [ModuleList(metrics) for metrics in per_split_metrics]
        self.loss = loss

        if to_probabilities == 'sigmoid':
            self.to_probabilities = Sigmoid()
        elif to_probabilities == 'softmax':
            self.to_probabilities = Softmax(dim=-1)

    def forward(self, loop, y_pred, y_true, split: DatasetSplit):
        y_prob = self.to_probabilities(y_pred)

        if split == DatasetSplit.TRAIN:
            metrics = self.train_metrics
        elif split == DatasetSplit.TEST:
            metrics = self.test_metrics
        else:
            metrics = self.val_metrics

        for metric in metrics:
            metric(y_prob, y_true.long())
            loop.log(f'{split.value}/' + self.classname(metric),
                     metric,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(y_true))

        loss = self.loss(y_pred, y_true)
        loop.log(f'{split.value}/loss', loss, on_step=False, on_epoch=True, batch_size=len(y_true))

        if split == DatasetSplit.TRAIN:
            return loss

    def metric_log(self, loop, y_pred, y_true, split: DatasetSplit):
        return self.forward(loop, y_pred, y_true, split)

    @staticmethod
    def classname(obj, lower=True):
        """Get the classname of an object.

        :param obj: any python object.
        :param lower: return the name in lowercase.
        :return: the classname as string.
        """
        name = obj.__class__.__name__
        return name.lower() if lower else name
