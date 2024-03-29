import csv
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from random import Random

import h5py
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset, Subset

from datamodule import DatasetSplit
from datamodule.dataset_utils import split_dataset, min_max_normalize, export_splits
from logger.utils import get_logger
from preprocessor import Preprocessor


class TrainValTestDataset(Dataset, ABC):
    def __init__(self, train_metrics, val_metrics, test_metrics):
        self.train_metrics = [str(m) for m in train_metrics]
        self.val_metrics = [str(m) for m in val_metrics]
        self.test_metrics = [str(m) for m in test_metrics]

    @abstractmethod
    def get_split(self, split: DatasetSplit) -> Dataset:
        """

        """

    @abstractmethod
    def prepare_data(self):
        """

        """


class JSONLDataset(TrainValTestDataset):
    """A dataset that reads dict instances from a jsonl file into memory, and applies preprocessing to each instance.
    """

    def __init__(self, name, task, train_file, val_file, test_file, preprocessor: Preprocessor, num_train_samples,
                 num_test_samples, num_classes, train_metrics, val_metrics, test_metrics,
                 labels_to_onehot: bool = False, raw_file=None, label_file=None):
        """

        :param task: name of the task the dataset will be used for.
        :param preprocessor: preprocessor to be applied to each instance.
        """

        super().__init__(train_metrics, val_metrics, test_metrics)
        self.log = get_logger(self)

        self.name = name
        self.task = task

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.num_classes = num_classes

        self.raw_file = raw_file

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.label_file = label_file

        self.preprocessor = preprocessor

        self._label2id = None
        self._labels_to_onehot = labels_to_onehot

        self.instances = None

    def __getitem__(self, index):
        x = self.instances[index]

        spans, labels = self._unpack_inputs(x)
        subject_in, new_spans, new_labels = self.preprocessor(x['text'], spans, labels)

        return subject_in, new_spans, new_labels

    def __len__(self):
        return len(self.instances)

    @staticmethod
    def read_labels(label_file):
        with open(to_absolute_path(label_file), 'r') as fp:
            label2id = {x.rstrip(): i for i, x in enumerate(fp)}
        return label2id

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
        if self.label_file is not None:
            self._label2id = self.read_labels(self.label_file)
            # replace label strings with integer ids or one-hot encoding for training
            one_hot_lookup = torch.eye(len(self._label2id))
            for instance in instances:
                for target in instance['targets']:
                    label_str = str(target['label'])
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

    def get_split(self, split: DatasetSplit) -> Dataset:
        self._init_instances(split)
        return deepcopy(self)

    def prepare_data(self):
        if self.raw_file is not None:
            raw_dataset = to_absolute_path(self.raw_file)
            to_be_generated = list(map(to_absolute_path,
                                       [self.train_file, self.val_file, self.test_file]))

            if not all(map(os.path.exists, to_be_generated)):
                self.log.info('Generating dataset splits')
                # load raw data
                with open(raw_dataset, 'r', encoding='utf8') as fp:
                    ds = json.load(fp)

                # create random splits
                train_ds, val_ds, test_ds = split_dataset(ds,
                                                          self.num_train_samples,
                                                          self.num_test_samples)

                # normalize target scores between 0 and 1
                if self.task.normalize_target:
                    ds_splits, min_score_train, max_score_train = min_max_normalize(train_ds, val_ds, test_ds)
                    labels = None
                else:
                    ds_splits, min_score_train, max_score_train = (train_ds, val_ds, test_ds), None, None
                    labels = set(chain.from_iterable([[str(t['label']) for t in x['targets']] for x in ds]))

                # export splits as jsonl files
                output_dir = to_absolute_path(os.path.split(to_be_generated[0])[0])
                export_splits(output_dir, ds_splits, labels, min_score_train, max_score_train)

    def collate(self, data):
        return self.preprocessor.collate(data)


class RankingDataset(TrainValTestDataset):
    def __init__(self, name, task, data_file, train_file, val_file, test_file, qrels_file, preprocessor: Preprocessor,
                 num_classes, train_metrics, val_metrics, test_metrics, limit_train_samples=None):

        super().__init__(train_metrics, val_metrics, test_metrics)

        self.name = name
        self.task = task
        self.num_classes = num_classes
        self.limit_train_samples = limit_train_samples

        self._data_file = to_absolute_path(data_file)

        self._train_file = to_absolute_path(train_file)
        self._val_file = to_absolute_path(val_file)
        self._test_file = to_absolute_path(test_file)

        self._qrels_file = to_absolute_path(qrels_file)

        self.qid_to_pid_to_label = self._read_trec_qrels(self._qrels_file)
        self.current_file = None
        self.preprocessor = preprocessor

        self.split = None

    def __getitem__(self, index):
        with h5py.File(self.current_file, "r") as fp:
            q_id = fp["q_ids"][index]
            doc_id = fp["doc_ids"][index]

            og_q_id = self.get_original_query_id(q_id)
            og_doc_id = self.get_original_document_id(doc_id)

            if self.split == DatasetSplit.TEST:
                # for the test set we use the qrels file
                label = self.qid_to_pid_to_label[int(og_q_id)].get(int(og_doc_id), 0)
                label = torch.as_tensor([label])
            else:
                label = torch.tensor(fp["labels"][index]).unsqueeze(0).long()

        with h5py.File(self._data_file, "r") as fp:
            query = fp["queries"].asstr()[q_id]
            doc = fp["docs"].asstr()[doc_id]

        input_text = query + ' [SEP] ' + doc
        subject_in, new_spans, new_labels = self.preprocessor(input_text, labels=label)

        return og_q_id, og_doc_id, subject_in, new_spans, new_labels

    def __len__(self):
        with h5py.File(self.current_file, "r") as fp:
            return len(fp["q_ids"])

    def get_split(self, split: DatasetSplit) -> Dataset:
        if split == DatasetSplit.TRAIN:
            self.current_file = self._train_file

            if self.limit_train_samples is not None:
                training_samples = Random(5823905).sample(range(len(self)), k=self.limit_train_samples)
                return deepcopy(Subset(self, indices=training_samples))

        elif split == DatasetSplit.VALIDATION:
            self.current_file = self._val_file
        elif split == DatasetSplit.TEST:
            self.current_file = self._test_file
        else:
            raise NotImplementedError()

        self.split = split
        return deepcopy(self)

    def prepare_data(self):
        pass

    def collate(self, data):
        q_ids, doc_ids, text_pairs, spans, labels = zip(*data)
        (encodings, spans), labels = self.preprocessor.collate(zip(text_pairs, spans, labels))

        q_ids = torch.tensor(q_ids, dtype=torch.long)
        doc_ids = torch.tensor(doc_ids, dtype=torch.long)

        rank_labels = labels
        binary_labels = torch.where(labels > 0, 1, 0)
        return q_ids, doc_ids, (encodings, spans), binary_labels, rank_labels

    def get_original_query_id(self, q_id: int):
        with h5py.File(self._data_file, "r") as fp:
            return int(fp["orig_q_ids"].asstr()[q_id])

    def get_original_document_id(self, doc_id: int):
        with h5py.File(self._data_file, "r") as fp:
            return int(fp["orig_doc_ids"].asstr()[doc_id])

    @staticmethod
    def _read_trec_qrels(qrels_file):
        qrels = defaultdict(dict)
        with open(qrels_file, 'r') as fp:
            for q_id, _, p_id, label in csv.reader(fp, delimiter=' '):
                q_id, p_id, label = map(int, [q_id, p_id, label])
                qrels[q_id][p_id] = int(label)

        return qrels
