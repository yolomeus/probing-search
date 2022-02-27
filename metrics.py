from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, RetrievalPrecision, RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG


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

class CustomRetrievalMixin:
    """Allows applying a torchmetrics RetrievalMetric to accept 2D predictions, by taking the second
    dimension as relevance score.
    """

    def update(self, preds: Tensor, target: Tensor, indexes: Tensor) -> None:
        super().update(preds[:, -1], target, indexes)


class MAP(CustomRetrievalMixin, RetrievalMAP):
    """ """


class MRR(CustomRetrievalMixin, RetrievalMRR):
    """ """


class PrecisionAt10(CustomRetrievalMixin, RetrievalPrecision):
    def __init__(self):
        super().__init__(k=10)


class PrecisionAt20(CustomRetrievalMixin, RetrievalPrecision):
    def __init__(self):
        super().__init__(k=20)


class NDCGAt10(CustomRetrievalMixin, RetrievalNormalizedDCG):
    def __init__(self):
        super().__init__(k=10)


class NDCGAt20(CustomRetrievalMixin, RetrievalNormalizedDCG):
    def __init__(self):
        super().__init__(k=20)
