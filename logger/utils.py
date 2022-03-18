"""Training loop related utilities.
"""
import csv
from logging import getLogger
from pathlib import Path
from typing import List, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module, ModuleList, Softmax, Sigmoid, Identity
from torchmetrics.retrieval import RetrievalMetric

from datamodule import DatasetSplit


class Metrics(Module):
    """Stores and manages metrics during training/testing for log.
    """

    def __init__(self, loss: Module, metrics_configs: List[DictConfig], to_probabilities: str):
        """

        :param loss: the loss module for computing the loss metric.
        :param metrics_configs: dict configs for each metric to instantiate.
        :param to_probabilities: either 'sigmoid', 'softmax' or None for Identity, used to convert raw model predictions into
        probabilities before passing them into a metric function.
        """
        super().__init__()

        per_split_metrics = [
            [] if metrics_configs is None else [instantiate(metric, compute_on_step=False) for metric in
                                                metrics_configs]
            for _ in range(3)]
        self.train_metrics, self.val_metrics, self.test_metrics = [ModuleList(metrics) for metrics in per_split_metrics]
        self.loss = loss

        if to_probabilities == 'sigmoid':
            self.to_probabilities = Sigmoid()
        elif to_probabilities == 'softmax':
            self.to_probabilities = Softmax(dim=-1)
        elif to_probabilities is None:
            self.to_probabilities = Identity()

    def forward(self, loop, y_pred, y_true, split: DatasetSplit, **kwargs):
        y_prob = self.to_probabilities(y_pred)

        if split == DatasetSplit.TRAIN:
            metrics = self.train_metrics
        elif split == DatasetSplit.TEST:
            metrics = self.test_metrics
        else:
            metrics = self.val_metrics

        for metric in metrics:
            # pass kwargs if accepted by metric
            if issubclass(type(metric), RetrievalMetric):
                metric(y_prob, y_true, **kwargs)
            else:
                metric(y_prob, y_true)

            loop.log(f'{split.value}/' + self.classname(metric),
                     metric,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(y_true))

        loss = self.loss(y_pred, y_true)
        loop.log(f'{split.value}/loss', loss, on_step=False, on_epoch=True, batch_size=len(y_true))

        if split == DatasetSplit.TRAIN:
            return loss

    def metric_log(self, loop, y_pred, y_true, split: DatasetSplit, **kwargs):
        return self.forward(loop, y_pred, y_true, split, **kwargs)

    @staticmethod
    def classname(obj, lower=True):
        """Get the classname of an object.

        :param obj: any python object.
        :param lower: return the name in lowercase.
        :return: the classname as string.
        """
        name = obj.__class__.__name__
        return name.lower() if lower else name


def get_logger(obj):
    """Get a python logging logger for an object containing its module and class name when logging.
    """
    return getLogger('.'.join([obj.__module__, obj.__class__.__name__]))


def write_trec_eval_file(out_file: Path, predictions: Dict[str, Dict[str, float]], name: str):
    """Write the results in a file accepted by the TREC evaluation tool.

    :param out_file: File to create
    :param predictions: Query IDs mapped to document IDs mapped to scores
    :param name: Method name
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        for q_id in predictions:
            ranking = sorted(
                predictions[q_id].keys(), key=predictions[q_id].get, reverse=True
            )
            for rank, doc_id in enumerate(ranking, 1):
                score = predictions[q_id][doc_id]
                writer.writerow([q_id, "Q0", doc_id, rank, score, name])
