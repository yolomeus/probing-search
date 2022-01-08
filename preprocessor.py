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
    def preprocess(self, input_text, spans=None, labels=None):
        """Preprocess input_text in such a way, that a subject model will accept it.

        :param input_text: the input text that will be fed to a subject model.
        :param spans: the target spans to adjust for the new tokenization if necessary.
        :param labels: labels for the processed text.
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


class EdgeProbingPreprocessor(Preprocessor, ABC):
    """Already collates spans and labels from edge probing inputs by flattening them and prepending batch indexes to
    each span. Needs to overwrite `text_collate` for collating text encodings depending on preprocessing.
    """

    def __init__(self, pair_targets: bool):
        self.pair_targets = pair_targets

    def collate(self, data):
        encodings, spans, labels = zip(*data)

        # prepend batch idx to each span and flatten
        if not self.pair_targets:
            # single span per prediction
            span_ids = self._spans_to_triples(spans)
        else:
            # a pair of spans per prediction
            spans1, spans2 = zip(*spans)
            span_ids = self._spans_to_triples(spans1), self._spans_to_triples(spans2)

        labels = torch.stack(tuple(chain(*labels)))
        batch_enc = self.text_collate(encodings)

        return (batch_enc, span_ids), labels

    @staticmethod
    def _spans_to_triples(spans):
        return np.concatenate([[(i, *span) for span in target_spans]
                               for i, target_spans in enumerate(spans)])

    @abstractmethod
    def text_collate(self, encodings):
        """Build a batch of inputs from multiple text encodings.
        :param encodings: a list of encoded texts.
        """


class BERTPreprocessor(EdgeProbingPreprocessor):
    def __init__(self, tokenizer_name, pair_targets: bool):
        super().__init__(pair_targets)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def preprocess(self, input_text, spans=None, labels=None):
        """
        :param spans: target spans from original tokenization
        :param input_text: the text to be fed into the subject model.
        :return: the subject model input representation.
        """

        if self.pair_targets:
            new_spans1 = self._retokenize_spans(input_text, spans[0])
            new_spans2 = self._retokenize_spans(input_text, spans[1])
            new_spans = (new_spans1, new_spans2)
        else:
            new_spans = self._retokenize_spans(input_text, spans)

        tokens_full = self.tokenizer(input_text, truncation=True)
        return tokens_full, new_spans, labels

    def _retokenize_spans(self, original_text, spans):
        """Given a list of original tokens and spans, recompute new spans for the wordpiece tokenizer.
        
        :param original_text: the original text, where tokens are separated by whitespace.
        :param spans: the original spans w.r.t. the original tokens.
        :return: 
        """

        vanilla_tokens = original_text.split()
        left_spans = [' '.join(vanilla_tokens[:a]) for a, _ in spans]
        original_spans = [' '.join(vanilla_tokens[a:b]) for a, b in spans]

        left_retokenized = [self.tokenizer(left_span,
                                           add_special_tokens=False)['input_ids']
                            for left_span in left_spans]
        span_retokenized = [self.tokenizer(span,
                                           add_special_tokens=False)['input_ids']
                            for span in original_spans]

        # shift 1 by one to account for the [cls] token in the beginning
        new_spans = [(1 + len(a), 1 + len(a) + len(b))
                     for a, b in zip(left_retokenized, span_retokenized)]
        return new_spans

    def text_collate(self, encodings):
        batch_enc = BatchEncoding({'input_ids': [],
                                   'token_type_ids': [],
                                   'attention_mask': []})

        for enc in encodings:
            batch_enc['input_ids'].append(enc['input_ids'])
            batch_enc['token_type_ids'].append(enc['token_type_ids'])
            batch_enc['attention_mask'].append(enc['attention_mask'])

        batch_enc = self.tokenizer.pad(batch_enc, return_tensors='pt')
        return batch_enc
