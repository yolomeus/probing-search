import json

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datamodule import DatasetSplit
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

        self._label2id = self.read_labels(self._dataset_conf.label_file)

    @property
    def train_ds(self):
        return JSONLDataset(self._dataset_conf.train_file, self._label2id, self._preprocessor)

    @property
    def val_ds(self):
        return JSONLDataset(self._dataset_conf.dev_file, self._label2id, self._preprocessor)

    @property
    def test_ds(self):
        return JSONLDataset(self._dataset_conf.test_file, self._label2id, self._preprocessor)

    @staticmethod
    def read_labels(label_file):
        with open(to_absolute_path(label_file), 'r') as fp:
            label2id = {x.rstrip(): i for i, x in enumerate(fp)}
        return label2id


class JSONLDataset(Dataset):
    """A dataset that reads dict instances from a jsonl file into memory, and applies preprocessing to each instance.
    """

    def __init__(self, filepath, label2id, preprocessor: Preprocessor):
        """

        :param filepath: path to the jsonl file to load instances from.
        """
        self._preprocessor = preprocessor
        with open(to_absolute_path(filepath), 'r') as fp:
            self.instances = tuple(json.loads(line) for line in fp)

        # replace label strings with integer ids for training
        for instance in self.instances:
            for target in instance['targets']:
                label_str = target['label']
                target['label'] = label2id[label_str]

    def __getitem__(self, index):
        x = self.instances[index]

        spans, labels = zip(*[(t['span1'], t['label']) for t in x['targets']])
        subject_in = self._preprocessor(x['text'])

        return subject_in, spans, labels

    def __len__(self):
        return len(self.instances)
