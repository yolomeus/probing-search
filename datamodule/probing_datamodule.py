import json
import os.path
import pickle

import torch
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
        :param preprocessor: the preprocessor to apply to each instance and use for batch collation.
        """
        super().__init__(*args, **kwargs)
        self._dataset_conf = dataset
        self._preprocessor = preprocessor

        self._label2id = self.read_labels(self._dataset_conf.label_file)

    @property
    def train_ds(self):
        return JSONLDataset(self._dataset_conf.task,
                            self._dataset_conf.train_file,
                            self._label2id,
                            self._preprocessor)

    @property
    def val_ds(self):
        return JSONLDataset(self._dataset_conf.task,
                            self._dataset_conf.dev_file,
                            self._label2id,
                            self._preprocessor)

    @property
    def test_ds(self):
        return JSONLDataset(self._dataset_conf.task,
                            self._dataset_conf.test_file,
                            self._label2id,
                            self._preprocessor)

    @staticmethod
    def read_labels(label_file):
        with open(to_absolute_path(label_file), 'r') as fp:
            label2id = {x.rstrip(): i for i, x in enumerate(fp)}
        return label2id

    def build_collate_fn(self, split: DatasetSplit = None):
        # the preprocessor decides how to collate a batch as it knows what a single instance looks like.
        return self._preprocessor.collate


class JSONLDataset(Dataset):
    """A dataset that reads dict instances from a jsonl file into memory, and applies preprocessing to each instance.
    """

    def __init__(self, task, filepath, label2id, preprocessor: Preprocessor):
        """

        :param task: name of the task the dataset will be used for.
        :param label2id: a mapping from string labels to integer ids.
        :param preprocessor: preprocessor to be applied to each instance.
        :param filepath: path to the jsonl file to load instances from.
        """

        self._task = task
        self._filepath = filepath
        self._preprocessor = preprocessor
        self._label2id = label2id

        self._one_hot_lookup = torch.eye(len(label2id))
        self.instances = self._init_instances()

    def _init_instances(self):
        """Read instances, convert labels to one-hot encodings and exclude examples with no targets if needed for task.

        :return: a tuple of edge probing instances.
        """
        with open(to_absolute_path(self._filepath), 'r') as fp:
            instances = tuple(json.loads(line) for line in fp)

        if self._task.requires_target_spans:
            # filter out instances with no target spans
            instances = tuple(filter(lambda x: len(x['targets']) > 0, instances))

        # replace label strings with integer ids for training
        for instance in instances:
            for target in instance['targets']:
                label_str = target['label']
                label_id = self._label2id[label_str]
                target['label'] = self._one_hot_lookup[label_id]

        return instances

    def __getitem__(self, index):
        x = self.instances[index]

        if self._task.has_pair_targets:
            spans1, spans2, labels = zip(*[(t['span1'], t['span2'], t['label'])
                                           for t in x['targets']])
            spans = (spans1, spans2)
        else:
            spans, labels = zip(*[(t['span1'], t['label']) for t in x['targets']])

        subject_in, new_spans = self._preprocessor(x['text'], spans)
        return subject_in, new_spans, labels

    def __len__(self):
        return len(self.instances)
