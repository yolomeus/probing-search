from abc import abstractmethod

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

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

    @property
    @abstractmethod
    def train_ds(self):
        """Build the train pytorch dataset.

        :return: the train pytorch dataset.
        """

    @property
    @abstractmethod
    def val_ds(self):
        """Build the validation pytorch dataset.

        :return: the validation pytorch dataset.
        """

    @property
    @abstractmethod
    def test_ds(self):
        """Build the test pytorch dataset.

        :return: the test pytorch dataset.
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
