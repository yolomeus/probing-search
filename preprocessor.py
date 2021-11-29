from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from transformers import BertTokenizer, BatchEncoding


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
        return default_collate(data)

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)


class BERTPreprocessor(Preprocessor):
    def __init__(self, tokenizer_name):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def preprocess(self, input_text):
        """
        :param input_text: the text to be fed into the subject model.
        :return: the subject model input representation
        """
        tokens = self.tokenizer(input_text, truncation=True)
        return tokens

    def collate(self, data):
        encodings, spans, labels = zip(*data)
        # prepend batch idx to each span and flatten
        span_ids = np.concatenate([[(i, *span) for span in target_spans]
                                   for i, target_spans in enumerate(spans)])
        labels = torch.stack(tuple(chain(*labels)))

        batch_enc = BatchEncoding({'input_ids': [],
                                   'token_type_ids': [],
                                   'attention_mask': []})

        for enc in encodings:
            batch_enc['input_ids'].append(enc['input_ids'])
            batch_enc['token_type_ids'].append(enc['token_type_ids'])
            batch_enc['attention_mask'].append(enc['attention_mask'])

        batch_enc = self.tokenizer.pad(batch_enc, return_tensors='pt')
        return (batch_enc, span_ids), labels
