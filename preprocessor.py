from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """Process a instance in the edge probing format
    """

    @abstractmethod
    def preprocess(self, edge_probe):
        """
        :param edge_probe: a single dict in the edge probing format.
        :return: the preprocessed instance, e.g. tokenized sequence + labels
        """
        pass


class DummyPreprocessor(Preprocessor):
    def preprocess(self, edge_probe):
        return [ord(c) for c in edge_probe['text']], ['targets']
