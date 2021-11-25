from torch.nn import Module
from transformers import BertModel


class BERTPretrained(Module):
    """A Pre-Trained BERT from the huggingface transformers library that returns hidden states of all layers, including
    the initial matrix embeddings (layer 0).
    """

    def __init__(self, model_name):
        """

        :param model_name: huggingface model name or path to model checkpoint file
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs, output_hidden_states=True)
        return outputs.hidden_states
