import json

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datamodule.default_datamodule import AbstractDefaultDataModule
from preprocessor import Preprocessor


class ProbingDataModule(AbstractDefaultDataModule):
    """DataModule for datasets in the edge probing format stored as jsonl files.
    """

    def __init__(self, dataset: DictConfig, preprocessor: Preprocessor, *args, **kwargs):
        """

        :param dataset: config with attributes `train_file`, `dev_file`, `test_file` which are paths to jsonl files.
        :param preprocessor:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self._dataset_conf = dataset
        self._preprocessor = preprocessor

    @property
    def train_ds(self):
        return JSONLDataset(self._dataset_conf.train_file, self._preprocessor)

    @property
    def val_ds(self):
        return JSONLDataset(self._dataset_conf.dev_file, self._preprocessor)

    @property
    def test_ds(self):
        return JSONLDataset(self._dataset_conf.test_file, self._preprocessor)


class JSONLDataset(Dataset):
    """A dataset that reads dict instances from a jsonl file into memory, and applies preprocessing to each instance.
    """

    def __init__(self, filepath, preprocessor: Preprocessor):
        """

        :param filepath: path to the jsonl file to load instances from.
        :param preprocessor: the preprocessors which is applied to each instance.
        """
        self._preprocessor = preprocessor
        with open(to_absolute_path(filepath), 'r') as fp:
            self.x = tuple(json.loads(line) for line in fp)

    def __getitem__(self, index):
        # TODO depending on ds size we might want to cache preprocessed instances in memory
        x, y = self._preprocessor.preprocess(self.x[index])
        return x, y

    def __len__(self):
        return len(self.x)
