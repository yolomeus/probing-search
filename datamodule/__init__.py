from enum import Enum


class DatasetSplit(Enum):
    """Enum for train, validation and test split.
    """
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'
