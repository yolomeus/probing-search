from abc import ABC
from typing import Any

import torch
from torch import Tensor, tensor
from torch.nn import functional as F
from torchmetrics import Metric, RetrievalPrecision, RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG
from torchmetrics.retrieval import RetrievalMetric
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import get_group_indexes


class MDL(Metric):
    """The minimum description length of a probing model, when computed through the online-coding approach.
    """

    def __init__(self, num_portions: int, num_classes: int):
        super().__init__()
        self.first_portion_size = None
        self.num_classes = torch.tensor(num_classes, dtype=torch.float32)

        for i in range(num_portions):
            self.add_state(f'losses', default=torch.zeros((num_portions,)), dist_reduce_fx='sum')

    def update(self, preds, target, portion_idx, first_portion_size=None) -> None:
        """

        :param preds: batch of predictions on data chunk i+1, i.e. (portion(i + 1) - portion(i)).
        :param target: batch of targets on on data chunk i+1, i.e. (portion(i + 1) - portion(i)).
        :param portion_idx: index of the current online portion for which `preds` are passed.
        :param first_portion_size: number of targets in the first training portion
        """
        self.losses[portion_idx] += F.cross_entropy(preds, target, reduction='sum')

        if self.first_portion_size is None:
            self.first_portion_size = first_portion_size

    def compute(self) -> Any:
        sum_left = self.first_portion_size * torch.log2(self.num_classes)
        sum_right = self.losses.sum()

        return sum_left + sum_right


class Compression(Metric):
    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = torch.tensor(num_classes, dtype=torch.float32)
        self.add_state('mdl', torch.zeros((1, 1)))
        self.add_state('num_targets', torch.zeros((1, 1)))

    def update(self, mdl, num_train_targets):
        self.mdl[0] = mdl
        self.num_targets[0] = num_train_targets

    def compute(self) -> Any:
        uniform_codelength = self.num_targets * torch.log2(self.num_classes)
        return uniform_codelength / self.mdl


# ----- Retrieval ----- #


class CustomRetrievalMixin(RetrievalMetric, ABC):
    """Mixin that can be used on subclasses of `RetrievalMetric` to change the way state is tracked. Unlike
    `RetrievalMetric` this implementation doesn't use lists for tracking state which caused massive slowdown when
    tracking large amounts of predictions.
    Note that this implementation stores large tensors instead and might cause higher gpu memory usage.
    """

    indexes: Tensor
    preds: Tensor
    target: Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # replace original state variables
        self.add_state("indexes", default=torch.empty(0, dtype=torch.int32), dist_reduce_fx=None)
        self.add_state("preds", default=torch.empty(0, dtype=torch.float32), dist_reduce_fx=None)
        self.add_state("target", default=torch.empty(0, dtype=torch.int32), dist_reduce_fx=None)
        self.add_state("idx", default=torch.zeros((1,), dtype=torch.long))

    def update(self, preds: Tensor, target: Tensor, indexes: Tensor) -> None:
        # we expect 2D predictions with the second dimension representing a relevance score.
        preds = preds[:, -1]

        if indexes is None:
            raise ValueError("Argument `indexes` cannot be None")

        indexes, preds, target = _check_retrieval_inputs(
            indexes, preds, target, allow_non_binary_target=self.allow_non_binary_target, ignore_index=self.ignore_index
        )

        # instead of appending to python lists, we concatenate to our state tensors.
        self.indexes = torch.cat([self.indexes, indexes.to(torch.int32)])
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target.to(torch.int32)])

        self.idx += len(indexes)

    def compute(self) -> Tensor:
        # this is identical to `RetrievalMetric`´s compute except that we do not have to concatenate the python lists
        # as a first step
        indexes = self.indexes
        preds = self.preds
        target = self.target

        res = []
        groups = get_group_indexes(indexes)

        for group in groups:
            mini_preds = preds[group]
            mini_target = target[group]

            if not mini_target.sum():
                if self.empty_target_action == "error":
                    raise ValueError("`compute` method was provided with a query with no positive target.")
                if self.empty_target_action == "pos":
                    res.append(tensor(1.0))
                elif self.empty_target_action == "neg":
                    res.append(tensor(0.0))
            else:
                # ensure list contains only float tensors
                res.append(self._metric(mini_preds, mini_target))

        return torch.stack([x.to(preds) for x in res]).mean() if res else tensor(0.0).to(preds)


class MAP(CustomRetrievalMixin, RetrievalMAP):
    """Mean Average Precision"""


class MRR(CustomRetrievalMixin, RetrievalMRR):
    """Mean Reciprocal Rank"""


class PrecisionAt10(CustomRetrievalMixin, RetrievalPrecision):
    def __init__(self, **kwargs):
        super().__init__(k=10, **kwargs)


class PrecisionAt20(CustomRetrievalMixin, RetrievalPrecision):
    def __init__(self, **kwargs):
        super().__init__(k=20, **kwargs)


class NDCGAt10(CustomRetrievalMixin, RetrievalNormalizedDCG):
    def __init__(self, **kwargs):
        super().__init__(k=10, **kwargs)


class NDCGAt20(CustomRetrievalMixin, RetrievalNormalizedDCG):
    def __init__(self, **kwargs):
        super().__init__(k=20, **kwargs)
