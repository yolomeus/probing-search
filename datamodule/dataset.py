import json
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset

from datamodule import DatasetSplit
from preprocessor import Preprocessor


class TrainValTestDataset(Dataset, ABC):
    @abstractmethod
    def get_split(self, split: DatasetSplit) -> Dataset:
        """

        """


class JSONLDataset(TrainValTestDataset):
    """A dataset that reads dict instances from a jsonl file into memory, and applies preprocessing to each instance.
    """

    def __init__(self,
                 name,
                 task,
                 train_file,
                 val_file,
                 test_file,
                 preprocessor: Preprocessor,
                 labels_to_onehot: bool,
                 num_train_samples,
                 num_test_samples,
                 num_classes,
                 raw_file=None,
                 label2id=None):
        """

        :param task: name of the task the dataset will be used for.
        :param label2id: a mapping from string labels to integer ids.
        :param preprocessor: preprocessor to be applied to each instance.
        :param filepath: path to the jsonl file to load instances from.
        """

        self.name = name
        self.task = task

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.num_classes = num_classes

        self.raw_file = raw_file

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        self.preprocessor = preprocessor

        self._labels_to_onehot = labels_to_onehot
        self._label2id = label2id

        self.instances = None

    def get_split(self, split: DatasetSplit) -> Dataset:
        self._init_instances(split)
        return deepcopy(self)

    def __getitem__(self, index):
        x = self.instances[index]

        spans, labels = self._unpack_inputs(x)
        subject_in, new_spans, new_labels = self.preprocessor(x['text'], spans, labels)

        return subject_in, new_spans, new_labels

    def __len__(self):
        return len(self.instances)

    def _init_instances(self, split: DatasetSplit):
        """Read instances, convert labels to one-hot encodings and exclude examples with no targets if needed for task.

        :return: a tuple of edge probing instances.
        """

        if split == DatasetSplit.TRAIN:
            filepath = self.train_file
        elif split == DatasetSplit.VALIDATION:
            filepath = self.val_file
        elif split == DatasetSplit.TEST:
            filepath = self.test_file
        else:
            raise NotImplementedError()

        with open(to_absolute_path(filepath), 'r') as fp:
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

        self.instances = instances

    def _unpack_inputs(self, x):
        """Unpack spans and labels from an edge probing instance.

        :param x: the edge probing dict to unpack.
        :return: a tuple of (spans, labels), where spans is a pair of spans in case of span targets.
        """

        target_0 = x['targets'][0]
        if 'span1' not in target_0:
            # we will need to compute spans dynamically
            return None, [t['label'] for t in x['targets']]

        if not self.task.single_span:
            spans1, spans2, labels = zip(*[(t['span1'], t['span2'], t['label'])
                                           for t in x['targets']])
            spans = (spans1, spans2)
        else:
            spans, labels = zip(*[(t['span1'], t['label']) for t in x['targets']])

        return spans, labels
