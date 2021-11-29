import torch
from torch.nn import Module, Linear


class AttentionPooler(Module):
    """Attention pooling as described in https://arxiv.org/pdf/1905.06316.pdf (page 14, C).
    """

    def __init__(self, input_dim, hidden_dim, layer_to_probe):
        super().__init__()
        self.layer_to_probe = layer_to_probe
        self.in_projection = Linear(input_dim, hidden_dim)
        self.attention_scorer = Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, target_spans):
        """

        :param hidden_states: the hidden states of the subject model with shape (N, L, D), i.e. batch size,
        sequence length and hidden dimension.
        :param target_spans: a list of spans where each span is represented as triple: (i, a, b) with i being the
        span's position in the batch, and [a, b) the span interval along the sequence dimension.
        :return:
        """
        hidden_states = hidden_states[self.layer_to_probe]
        embed_spans = [hidden_states[i, a:b] for i, a, b in target_spans]

        embed_spans = [self.in_projection(span) for span in embed_spans]
        att_vectors = [self.attention_scorer(span).softmax(0)
                       for span in embed_spans]

        pooled_spans = [att_vec.T @ embed_span
                        for att_vec, embed_span in zip(embed_spans, att_vectors)]

        return torch.stack(pooled_spans).squeeze()
