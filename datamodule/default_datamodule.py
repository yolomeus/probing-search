from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from datamodule import DatasetSplit


class AbstractDefaultDataModule(LightningDataModule):
    """Base class for pytorch-lightning DataModule datasets. You can subclass this if you have a standard train,
    validation, test split."""

    def __init__(self, train_conf, test_conf, num_workers, pin_memory, persistent_workers=True):
        """
        :param train_conf: global training configuration
        :param test_conf: global test configuration
        :param num_workers: num_workers argument passed to all DataLoaders.
        :param pin_memory: pin_memory argument passed to all DataLoaders.
        :param persistent_workers: persistent_workers argument passed to all DataLoaders.
        """
        super().__init__()

        self._train_conf = train_conf
        self._test_conf = test_conf
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Assign train_ds, val_ds and test_ds here.
        """

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            self._train_conf.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.build_collate_fn(DatasetSplit.TRAIN),
            persistent_workers=self._persistent_workers)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            self.val_ds,
            self._test_conf.batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.build_collate_fn(DatasetSplit.VALIDATION),
            persistent_workers=self._persistent_workers
        )
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            self.test_ds,
            self._test_conf.batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.build_collate_fn(DatasetSplit.TEST),
            persistent_workers=self._persistent_workers
        )
        return test_dl

    # noinspection PyMethodMayBeStatic
    def build_collate_fn(self, split: DatasetSplit = None):
        """Override to define a custom collate function. Build a function for collating multiple data instances into
        a batch. Defaults to returning `None` since it's the default for DataLoader's collate_fn argument.

        While different collate functions might be needed depending on the dataset split, in most cases the same
        function can be returned for all data splits.

        :param split: The split that the collate function is used on to build batches. Can be ignored when train and
        test data share the same structure.
        :return: a single argument function that takes a list of tuples/instances and returns a batch as tensor or a
        tuple of multiple batch tensors.
        """

        return None


class MultiPortionMixin(AbstractDefaultDataModule, ABC):
    """DataModule for training on different portions, i.e. subsets of the train dataset. Can be used as Mixin to
    provide multi-portion functionality to a subclass of AbstractDefaultDataModule.
    """

    def __init__(self, portions: List, shuffle_first: bool = True, *args, **kwargs):
        """

         :param portions: A list of percentages which specify the size of each training portion to iterate over.
        :param shuffle_first: whether to shuffle the training set before dividing into portions.
        :param args: args passed to the parent class constructor.
        :param kwargs: kwargs passed to the parent class constructor.
        """
        super().__init__(*args, **kwargs)

        self.portion_idx = None
        self.portions = portions

        self._shuffle_first = shuffle_first
        self._training_ids = None

    def setup(self, stage=None):
        super().setup(stage)
        self._training_ids = self._init_training_ids()

    def _init_training_ids(self):
        n_train = len(self.train_ds)
        training_ids = torch.randperm(n_train) if self._shuffle_first else torch.arange(n_train)
        self.portion_idx = 0
        return training_ids

    def next_portion(self):
        """Change internal state to the next portion (as defined in `portions`), such that train_dataloader will return
        only the current portion of the train dataset.
        """
        if self.portion_idx < len(self.portions) - 1:
            self.portion_idx += 1
        else:
            raise OverflowError('already reached last portion')

    def train_dataloader(self):
        ds = self.train_ds
        if self.portion_idx is not None:
            ds = self._train_portion(ds)

        return DataLoader(
            ds,
            self._train_conf.batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.build_collate_fn(DatasetSplit.TRAIN),
            persistent_workers=self._persistent_workers
        )

    def _train_portion(self, ds):
        k = int(len(self._training_ids) * self.portions[self.portion_idx])
        return Subset(ds, indices=self._training_ids[:k])

    def portion_size(self, i):
        """Compute number of instances in the i-th portion in `portions`.

        :param i: portion to get the number of instances of.
        :return: number of instances in potion i.
        """
        return int(self.portions[i] * len(self.train_ds))

    @property
    def current_portion_size(self):
        """Number of instances in the currently selected portion.
        """
        return self.portion_size(self.portion_idx)

    @property
    def current_portion_percentage(self):
        """Current portion size as percentage.
        """
        return self.portions[self.portion_idx]
