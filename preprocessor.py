from abc import ABC, abstractmethod

import numpy as np
import spacy
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

        labels = torch.cat(labels, dim=0)
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

    def __init__(self, tokenizer_name, bucketize_labels: int, pair_targets: bool):
        super().__init__(pair_targets)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.bucketize_labels = bucketize_labels

    def preprocess(self, input_text, spans=None, labels=None):
        query, passage = input_text.split(' [SEP] ')

        labels = torch.tensor(labels)
        if self.bucketize_labels > 1:
            labels = self._bucketize_labels(labels)

        if spans is None:
            # we need to compute spans ourselves
            token_ids = self.tokenizer(query, passage, truncation=True, add_special_tokens=True)
            type_ids = torch.as_tensor(token_ids['token_type_ids'])

            query_len = len(type_ids[type_ids == 0]) - 1
            query_span = [1, query_len]
            passage_span = [query_len + 1, len(type_ids) - 1]

            return (query, passage), ([query_span], [passage_span]), labels

        return (query, passage), spans, labels

    def text_collate(self, encodings):
        queries, passages = zip(*encodings)
        batch_encoding = self.tokenizer(queries,
                                        passages,
                                        truncation=True,
                                        padding='longest',
                                        return_tensors='pt')
        return batch_encoding

    def _bucketize_labels(self, labels):
        boundaries = torch.linspace(0, 1, self.bucketize_labels)[1:]
        return torch.bucketize(labels, boundaries)


class BERTRetokenizationPreprocessor(EdgeProbingPreprocessor):
    """A preprocessor that expects pre-computed spans based on SpaCy tokenization. BERT tokenization is applied
    and the spans are recomputed accordingly.
    """

    def __init__(self, tokenizer_name, pair_targets: bool):
        super().__init__(pair_targets)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        spacy.cli.download('en_core_web_sm')
        self.spacy_tokenizer = spacy.load('en_core_web_sm').tokenizer

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

        tokens_full = self.tokenizer(*input_text.split(' [SEP] '), truncation=True)
        return tokens_full, new_spans, torch.tensor(labels)

    def _retokenize_spans(self, original_text, spans):
        """Given a list of original tokens and spans, recompute new spans for the wordpiece tokenizer.
        
        :param original_text: the original text, where tokens are separated by whitespace.
        :param spans: the original spans w.r.t. the original tokens.
        :return: 
        """

        query, passage = original_text.split(' [SEP] ')
        vanilla_tokens = [str(token) for token in self.spacy_tokenizer(query + ' ' + passage)]

        # everything left to the span
        text_left_spans = [' '.join(vanilla_tokens[:a]) for a, _ in spans]
        text_original_spans = [' '.join(vanilla_tokens[a:b]) for a, b in spans]

        left_retokenized = [self.tokenizer(left_span,
                                           add_special_tokens=False)['input_ids']
                            for left_span in text_left_spans]
        span_retokenized = [self.tokenizer(span,
                                           add_special_tokens=False)['input_ids']
                            for span in text_original_spans]

        # shift by one to account for the [cls] token in the beginning
        new_spans = []
        sep_pos = len(self.tokenizer(query, add_special_tokens=False))

        for a, b in zip(left_retokenized, span_retokenized):
            # shift by 1 for [cls]
            span = (1 + len(a), 1 + len(a) + len(b))
            # shift another for [sep] if span is within passage, i.e. right to sep
            if span[0] > sep_pos:
                span = (span[0] + 1, span[1] + 1)

            new_spans.append(span)

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
