"""Training loop related utilities.
"""
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module, ModuleList

from datamodule import DatasetSplit


class Metrics(Module):
    """Stores and manages metrics during training/testing for log.
    """

    def __init__(self, loss: Module, metrics_config: DictConfig):
        super().__init__()

        per_split_metrics = [[] if metrics_config is None else [instantiate(metric) for metric in metrics_config]
                             for _ in range(3)]
        self.train_metrics, self.val_metrics, self.test_metrics = [ModuleList(metrics) for metrics in per_split_metrics]
        self.loss = loss

    def forward(self, loop, y_pred, y_true, split: DatasetSplit):
        loss = self.loss(y_pred, y_true)
        loop.log(f'{split.value}/loss', loss, on_step=False, on_epoch=True, batch_size=len(y_true))

        y_prob = self._to_probabilities(y_pred)

        if split == DatasetSplit.TRAIN:
            metrics = self.train_metrics
        elif split == DatasetSplit.TEST:
            metrics = self.test_metrics
        else:
            metrics = self.val_metrics

        for metric in metrics:
            metric(y_prob.argmax(-1), y_true.long().argmax(-1))
            loop.log(f'{split.value}/' + self.classname(metric),
                     metric,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(y_true))

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

    @staticmethod
    def _to_probabilities(logits: Tensor):
        """Softmax normalize along the last dimension for multiclass targets (C>1), or use sigmoid in the case of 1D
        predictions (C=1).

        :param logits: a batch of raw, un-normalized prediction scores with shape (N, *, C).
        :return: a batch of probabilities.
        """
        return logits.sigmoid()
