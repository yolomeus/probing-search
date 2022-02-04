import json
from typing import List, Optional

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
            labels_to_onehot: bool,
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

        self._labels_to_onehot = labels_to_onehot

        if hasattr(self._dataset_conf, 'label_file'):
            self._label2id = self.read_labels(self._dataset_conf.label_file)
        else:
            self._label2id = None

    def setup(self, stage: Optional[str] = None) -> None:
        if None in [self.train_ds, self.val_ds, self.test_ds]:
            self.train_ds = self._build_ds(self._dataset_conf.train_file)
            self.val_ds = self._build_ds(self._dataset_conf.val_file)
            self.test_ds = self._build_ds(self._dataset_conf.test_file)

    def _build_ds(self, filepath):
        return JSONLDataset(
            self._dataset_conf.task,
            filepath,
            self._preprocessor,
            self._labels_to_onehot,
            self._label2id
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
            labels_to_onehot: bool,
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
            labels_to_onehot,
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
        return sum(map(lambda x: len(x[-1]), iter(self.train_ds)))

    @property
    def num_targets_portion(self):
        return sum(map(lambda x: len(x[-1]), iter(self._train_portion(self.train_ds))))


class JSONLDataset(Dataset):
    """A dataset that reads dict instances from a jsonl file into memory, and applies preprocessing to each instance.
    """

    def __init__(self, task, filepath, preprocessor: Preprocessor, labels_to_onehot: bool, label2id=None):
        """

        :param task: name of the task the dataset will be used for.
        :param label2id: a mapping from string labels to integer ids.
        :param preprocessor: preprocessor to be applied to each instance.
        :param filepath: path to the jsonl file to load instances from.
        """

        self._task = task
        self._filepath = filepath
        self._preprocessor = preprocessor
        self._labels_to_onehot = labels_to_onehot
        self._label2id = label2id

        self.instances = self._init_instances()

    def __getitem__(self, index):
        x = self.instances[index]

        spans, labels = self._unpack_inputs(x)
        subject_in, new_spans, new_labels = self._preprocessor(x['text'], spans, labels)

        return subject_in, new_spans, new_labels

    def __len__(self):
        return len(self.instances)

    def _init_instances(self):
        """Read instances, convert labels to one-hot encodings and exclude examples with no targets if needed for task.

        :return: a tuple of edge probing instances.
        """
        with open(to_absolute_path(self._filepath), 'r') as fp:
            instances = tuple(json.loads(line) for line in fp)

        # filter out instances with no target spans
        instances = tuple(filter(lambda x: len(x['targets']) > 0, instances))

        if self._label2id is not None:
            # replace label strings with integer ids or one-hot encoding for training
            one_hot_lookup = torch.eye(len(self._label2id))
            for instance in instances:
                for target in instance['targets']:
                    label_str = target['label']
                    label_id = self._label2id[label_str]
                    if self._labels_to_onehot:
                        target['label'] = one_hot_lookup[label_id]
                    else:
                        target['label'] = label_id

        return instances

    def _unpack_inputs(self, x):
        """Unpack spans and labels from an edge probing instance.

        :param x: the edge probing dict to unpack.
        :return: a tuple of (spans, labels), where spans is a pair of spans in case of span targets.
        """

        target_0 = x['targets'][0]
        if 'span1' not in target_0:
            # we will need to compute spans dynamically
            return None, [t['label'] for t in x['targets']]

        if self._task.has_pair_targets:
            spans1, spans2, labels = zip(*[(t['span1'], t['span2'], t['label'])
                                           for t in x['targets']])
            spans = (spans1, spans2)
        else:
            spans, labels = zip(*[(t['span1'], t['label']) for t in x['targets']])

        return spans, labels
