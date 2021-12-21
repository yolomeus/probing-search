from torch.nn import Module
from transformers import BertModel
from transformers.utils import logging


class BERTPretrained(Module):
    """A Pre-Trained BERT from the huggingface transformers library that returns hidden states of all layers, including
    the initial matrix embeddings (layer 0).
    """

    def __init__(self, model_name):
        """

        :param model_name: huggingface model name or path to model checkpoint file
        """
        super().__init__()
        logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(model_name)
        logging.set_verbosity_warning()

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs, output_hidden_states=True)
        return outputs.hidden_states
