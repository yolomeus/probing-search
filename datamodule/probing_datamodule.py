import json
import os.path
import random
from itertools import chain
from logging import getLogger
from random import shuffle
from typing import List, Optional

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from datamodule import DatasetSplit
from datamodule.dataset import JSONLDataset
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
        self.log = getLogger('.'.join([self.__module__, self.__class__.__name__]))
        super().__init__(train_conf, test_conf, num_workers, pin_memory)

        self._dataset = dataset
        self._preprocessor = preprocessor

        self._labels_to_onehot = labels_to_onehot
        self._label2id = None

    def setup(self, stage: Optional[str] = None) -> None:
        if hasattr(self._dataset, 'label_file') and self._label2id is None:
            self._label2id = self.read_labels(self._dataset.label_file)

        if None in [self.train_ds, self.val_ds, self.test_ds]:
            self.train_ds = self._build_ds(self._dataset.train_file)
            self.val_ds = self._build_ds(self._dataset.val_file)
            self.test_ds = self._build_ds(self._dataset.test_file)

    def _build_ds(self, filepath):
        return JSONLDataset(
            self._dataset.task,
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

    def prepare_data(self) -> None:
        raw_dataset = to_absolute_path(self._dataset.raw_file)
        to_be_generated = list(map(to_absolute_path,
                                   [self._dataset.train_file, self._dataset.val_file, self._dataset.test_file]))

        if not all(map(os.path.exists, to_be_generated)):
            self.log.info('Generating dataset splits')
            # load raw data
            with open(raw_dataset, 'r', encoding='utf8') as fp:
                ds = json.load(fp)

            # create random splits
            train_ds, val_ds, test_ds = self._split_dataset(ds,
                                                            self._dataset.num_train_samples,
                                                            self._dataset.num_test_samples)

            # normalize target scores between 0 and 1
            if self._dataset.task.normalize_target:
                ds_splits, min_score_train, max_score_train = self._min_max_normalize(train_ds, val_ds, test_ds)
                labels = None
            else:
                ds_splits, min_score_train, max_score_train = (train_ds, val_ds, test_ds), None, None
                labels = set(chain.from_iterable([[t['label'] for t in x['targets']] for x in ds]))

            # export splits as jsonl files
            output_dir = to_absolute_path(os.path.split(to_be_generated[0])[0])
            self._export_splits(output_dir, ds_splits, labels, min_score_train, max_score_train)

    @staticmethod
    def _split_dataset(dataset, num_train_samples, num_test_samples):
        """Split dataset into train, test and validation.

        :param dataset: a list of instances to sample from.
        :param num_train_samples: The total number of train samples.
        :param num_test_samples: The total number of test samples to split into validation and test set.
        :return: train_ds, val_ds, test_ds
        """
        assert len(dataset) >= num_train_samples + num_test_samples, 'too many train/test samples: dataset too small'

        # shuffle with local random seed for splitting data: keep the same dataset split if performing training
        # with a different random seed
        shuffle(dataset, random.Random(5823905).random)
        train_ds, val_ds = dataset[num_test_samples:], dataset[:num_test_samples]
        train_ds = train_ds[:num_train_samples]
        val_ds, test_ds = val_ds[len(val_ds) // 2:], val_ds[:len(val_ds) // 2]

        return train_ds, val_ds, test_ds

    @staticmethod
    def _min_max_normalize(train_ds, val_ds, test_ds, bounds=None):
        """Min-Max normalize target according to min and max values of the train dataset. Targets in the validaion and
        test set that are larger or smaller will be truncated accordingly before normalization.

        :param train_ds: list of dicts in the edge_probing format with a single score 'target' entry.
        :param val_ds: the validation split.
        :param test_ds: the test split.
        :param bounds: Minimum and maximum to normalize between. Will be inferred from the train dataset if not
        provided.

        :return: (train_ds, val_ds, test_ds), min_score_train, max_score_train
        """
        scores_train = [x['target'] for x in train_ds]
        if bounds is None:
            min_score_train, max_score_train = min(scores_train), max(scores_train)
        else:
            min_score_train, max_score_train = bounds

        for split in [train_ds, val_ds, test_ds]:
            for x in split:
                # truncate and normalised based to training data stats
                truncated_score = max(min_score_train, min(x['target'], max_score_train))
                normalized_score = (truncated_score - min_score_train) / (max_score_train - min_score_train)

                x['targets'] = [{'label': normalized_score}]
                del x['target']

        return (train_ds, val_ds, test_ds), min_score_train, max_score_train

    @staticmethod
    def _export_splits(output_dir, splits, labels=None, min_score=None, max_score=None):
        """Export each split into its own jsonl file.

        :param output_dir: the directory to export the files to.
        :param splits: train, validation and test split.
        :param min_score: min_score used for normalization to add as metadata.
        :param max_score: max_score used for normalization to add as metadata.
        :return:
        """
        for i, file in enumerate(['train.jsonl', 'validation.jsonl', 'test.jsonl']):
            with open(os.path.join(output_dir, file), 'w', encoding='utf8') as fp:
                fp.writelines(map(lambda x: json.dumps(x) + '\n', splits[i]))

        if not (min_score is None or max_score is None):
            with open(os.path.join(output_dir, 'meta.json'), 'w', encoding='utf8') as fp:
                meta = {'max_score': max_score, 'min_score': min_score}
                json.dump(meta, fp)

        if labels is not None:
            with open(os.path.join(output_dir, 'labels.txt'), 'w', encoding='utf8') as fp:
                fp.writelines(map(lambda x: x + '\n', labels))


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
