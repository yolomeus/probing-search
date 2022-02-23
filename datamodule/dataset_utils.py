import json
import os
from random import shuffle, Random


def split_dataset(dataset, num_train_samples, num_test_samples):
    """Split dataset into train, test and validation.

    :param dataset: a list of instances to sample from.
    :param num_train_samples: The total number of train samples.
    :param num_test_samples: The total number of test samples to split into validation and test set.
    :return: train_ds, val_ds, test_ds
    """
    assert len(dataset) >= num_train_samples + num_test_samples, 'too many train/test samples: dataset too small'

    # shuffle with local random seed for splitting data: keep the same dataset split if performing training
    # with a different random seed
    shuffle(dataset, Random(5823905).random)
    train_ds, val_ds = dataset[num_test_samples:], dataset[:num_test_samples]
    train_ds = train_ds[:num_train_samples]
    val_ds, test_ds = val_ds[len(val_ds) // 2:], val_ds[:len(val_ds) // 2]

    return train_ds, val_ds, test_ds


def min_max_normalize(train_ds, val_ds, test_ds, bounds=None):
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


def export_splits(output_dir, splits, labels=None, min_score=None, max_score=None):
    """Export each split into its own jsonl file.

    :param output_dir: the directory to export the files to.
    :param splits: train, validation and test split.
    :param labels: a set of possible classes. Will be exported to a labels.txt file to be used as lookup table.
    :param min_score: min_score used for normalization to add as metadata.
    :param max_score: max_score used for normalization to add as metadata.
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
