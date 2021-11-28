from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """Process an instance in the edge probing format and collate multiple into a batch.
    """

    @abstractmethod
    def preprocess(self, input_text):
        """Preprocess input_text in such a way, that a subject model will accept it.

        :param input_text: the input text that will be fed to a subject model.
        :return: the preprocessed text.
        """

    @abstractmethod
    def collate(self, data):
        """Override to specify how multiple examples should be collated into a batch, e.g. padding, truncation etc.

        :param data: the data to transform into a batch as returned by preprocess.
        :return: a batch or a tuple of batches
        """
        pass


class DummyPreprocessor(Preprocessor):
    def preprocess(self, edge_probe):
        return [ord(c) for c in edge_probe['text']], ['targets']
