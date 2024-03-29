from typing import List, Optional

from torch.utils.data import DataLoader, Subset

from datamodule import DatasetSplit
from datamodule.dataset import TrainValTestDataset
from datamodule.default_datamodule import AbstractDefaultDataModule, MultiPortionMixin
from logger.utils import get_logger


class ProbingDataModule(AbstractDefaultDataModule):
    """DataModule for datasets in the edge probing format stored as jsonl files.
    """

    def __init__(
            self,
            dataset: TrainValTestDataset,
            train_conf,
            test_conf,
            num_workers,
            pin_memory
    ):
        """

        :param dataset: config with attributes `train_file`, `dev_file`, `test_file` which are paths to jsonl files.
        :param train_conf: global training configuration
        :param test_conf: global test configuration
        :param num_workers: num_workers argument passed to all DataLoaders.
        :param pin_memory: pin_memory argument passed to all DataLoaders.
        """
        self.log = get_logger(self)
        super().__init__(train_conf, test_conf, num_workers, pin_memory)

        self._dataset = dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if None in [self.train_ds, self.val_ds, self.test_ds]:
            self.train_ds = self._dataset.get_split(DatasetSplit.TRAIN)
            self.val_ds = self._dataset.get_split(DatasetSplit.VALIDATION)
            self.test_ds = self._dataset.get_split(DatasetSplit.TEST)

    def build_collate_fn(self, split: DatasetSplit = None):
        # the preprocessor decides how to collate a batch as it knows what a single instance looks like.

        return self._dataset.collate

    def prepare_data(self) -> None:
        self._dataset.prepare_data()


class MDLProbingDataModule(MultiPortionMixin, ProbingDataModule):
    """Multi-Portion version of the ProbingDataModule with a predict_dataloader that returns the difference between
    current and next portion. This is needed for computing MDL.
    """

    def __init__(
            self,
            portions: List,
            shuffle_first: bool,
            dataset,
            train_conf,
            test_conf,
            num_workers,
            pin_memory
    ):
        """

        :param portions: A list of percentages which specify the size of each training portion to iterate over.
        :param shuffle_first: whether to shuffle the training set before dividing into portions.
        :param dataset: config with attributes `train_file`, `dev_file`, `test_file` which are paths to jsonl files.
        :param train_conf: global training configuration
        :param test_conf: global test configuration
        :param num_workers: num_workers argument passed to all DataLoaders.
        :param pin_memory: pin_memory argument passed to all DataLoaders.
        """
        super().__init__(
            portions,
            shuffle_first,
            dataset,
            train_conf,
            test_conf,
            num_workers,
            pin_memory
        )
        self._num_targets_total = None

    def predict_dataloader(self):
        pred_dl = DataLoader(
            self.pred_ds,
            self._train_conf.batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.build_collate_fn(DatasetSplit.TRAIN)
        )
        return pred_dl

    @property
    def pred_ds(self):
        ds = self.train_ds
        if self.portion_idx is not None:
            return self._pred_portion(ds)

        return ds

    def _pred_portion(self, ds):
        k = int(len(self._training_ids) * self.portions[self.portion_idx])
        k_next = int(len(self._training_ids) * self.portions[self.portion_idx + 1])
        return Subset(ds, indices=self._training_ids[k:k_next])

    @property
    def num_targets_total(self):
        if self._num_targets_total is None:
            self._num_targets_total = sum(map(lambda x: len(x[-1]), iter(self.train_ds)))
        return self._num_targets_total

    @property
    def num_targets_portion(self):
        return sum(map(lambda x: len(x[-1]), iter(self._train_portion(self.train_ds))))
