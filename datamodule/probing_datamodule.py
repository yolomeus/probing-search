import json
import os
import pickle
from typing import List

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, Subset

from datamodule import DatasetSplit
from datamodule.default_datamodule import AbstractDefaultDataModule, MultiPortionMixin
from preprocessor import Preprocessor


class ProbingDataModule(AbstractDefaultDataModule):
    """DataModule for datasets in the edge probing format stored as jsonl files.
    """

    def __init__(
            self,
            dataset: DictConfig,
            preprocessor: Preprocessor,
            train_conf,
            test_conf,
            num_workers,
            pin_memory
    ):
        """

        :param dataset: config with attributes `train_file`, `dev_file`, `test_file` which are paths to jsonl files.
        :param preprocessor: the preprocessor to apply to each instance and use for batch collation.

        :param train_conf: global training configuration
        :param test_conf: global test configuration
        :param num_workers: num_workers argument passed to all DataLoaders.
        :param pin_memory: pin_memory argument passed to all DataLoaders.
        """
        super().__init__(train_conf, test_conf, num_workers, pin_memory)

        self._dataset_conf = dataset
        self._preprocessor = preprocessor

        self._label2id = self.read_labels(self._dataset_conf.label_file)

    @property
    def train_ds(self):
        return JSONLDataset(
            self._dataset_conf.task,
            self._dataset_conf.train_file,
            self._label2id,
            self._preprocessor
        )

    @property
    def val_ds(self):
        return JSONLDataset(
            self._dataset_conf.task,
            self._dataset_conf.dev_file,
            self._label2id,
            self._preprocessor
        )

    @property
    def test_ds(self):
        return JSONLDataset(
            self._dataset_conf.task,
            self._dataset_conf.test_file,
            self._label2id,
            self._preprocessor
        )

    @staticmethod
    def read_labels(label_file):
        with open(to_absolute_path(label_file), 'r') as fp:
            label2id = {x.rstrip(): i for i, x in enumerate(fp)}
        return label2id

    def build_collate_fn(self, split: DatasetSplit = None):
        # the preprocessor decides how to collate a batch as it knows what a single instance looks like.
        return self._preprocessor.collate


class MDLProbingDataModule(MultiPortionMixin, ProbingDataModule):
    """Multi-Portion version of the ProbingDataModule with a predict_dataloader that returns the difference between
    current and next portion. This is needed for computing MDL.
    """

    def __init__(
            self,
            portions: List,
            shuffle_first: bool,
            dataset,
            preprocessor,
            train_conf,
            test_conf,
            num_workers,
            pin_memory
    ):
        """

        :param portions: A list of percentages which specify the size of each training portion to iterate over.
        :param shuffle_first: whether to shuffle the training set before dividing into portions.

        :param dataset: config with attributes `train_file`, `dev_file`, `test_file` which are paths to jsonl files.
        :param preprocessor: the preprocessor to apply to each instance and use for batch collation.

        :param train_conf: global training configuration
        :param test_conf: global test configuration
        :param num_workers: num_workers argument passed to all DataLoaders.
        :param pin_memory: pin_memory argument passed to all DataLoaders.
        """
        super().__init__(
            portions,
            shuffle_first,
            dataset,
            preprocessor,
            train_conf,
            test_conf,
            num_workers,
            pin_memory
        )

    def predict_dataloader(self):
        pred_dl = DataLoader(
            self.pred_ds,
            self._train_conf.batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self.build_collate_fn(DatasetSplit.TRAIN),
            persistent_workers=True
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
        return sum(map(lambda x: len(x[-1]), iter(self.train_ds)))

    @property
    def num_targets_portion(self):
        return sum(map(lambda x: len(x[-1]), iter(self._train_portion(self.train_ds))))


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

        self.instances = self._init_instances()

    def __getitem__(self, index):
        x = self.instances[index]

        spans, labels = self._unpack_inputs(x)
        subject_in, new_spans = self._preprocessor(x['text'], spans)

        return subject_in, new_spans, labels

    def __len__(self):
        return len(self.instances)

    def _init_instances(self):
        """Read instances, convert labels to one-hot encodings and exclude examples with no targets if needed for task.

        :return: a tuple of edge probing instances.
        """
        with open(to_absolute_path(self._filepath), 'r') as fp:
            instances = tuple(json.loads(line) for line in fp)

        if self._task.requires_target_spans:
            # filter out instances with no target spans
            instances = tuple(filter(lambda x: len(x['targets']) > 0, instances))

        # replace label strings with one-hot encoding of integer ids for training
        one_hot_lookup = torch.eye(len(self._label2id))
        for instance in instances:
            for target in instance['targets']:
                label_str = target['label']
                label_id = self._label2id[label_str]
                target['label'] = one_hot_lookup[label_id]

        return instances

    def _unpack_inputs(self, x):
        """Unpack spans ans labels from an edge probing instance.

        :param x: the edge probing dict to unpack.
        :return: a tuple of (spans, labels), where spans is a pair of spans in case of span targets.
        """
        if self._task.has_pair_targets:
            spans1, spans2, labels = zip(*[(t['span1'], t['span2'], t['label'])
                                           for t in x['targets']])
            spans = (spans1, spans2)
        else:
            spans, labels = zip(*[(t['span1'], t['label']) for t in x['targets']])

        return spans, labels


class CachedJSONLDataset(JSONLDataset):
    """A cached version of the JSONLDataset where, during the 1st access, each item is written to a pickle file and
    read from disk afterwards.
    """

    def __init__(self, task, filepath, label2id, preprocessor: Preprocessor):
        super().__init__(task, filepath, label2id, preprocessor)

        self.cache_dir = os.path.join('./.cache', self._filepath.split('.')[0])
        self.cache_dir = to_absolute_path(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def __getitem__(self, index):
        path = os.path.join(self.cache_dir, f'{index}.pkl')
        was_cached = os.path.exists(path)
        if was_cached:
            return self.uncache_item(path)

        item = super().__getitem__(index)
        if not was_cached:
            self.cache_item(item, path)

        return item

    @staticmethod
    def cache_item(item, path):
        """write item to pickle file.
        """
        with open(path, 'wb') as fp:
            pickle.dump(item, fp)

    @staticmethod
    def uncache_item(path):
        """Read item from pickle file.
        """
        with open(path, 'rb') as fp:
            return pickle.load(fp)
