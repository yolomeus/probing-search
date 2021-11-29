from torch.nn import Module, Sequential, Linear, Tanh, LayerNorm, Dropout


class MLP(Module):
    """The 2 layer MLP used by Tenney et al. in https://arxiv.org/pdf/1905.06316.pdf.

    https://github.com/nyu-mll/jiant/blob/ead63af002e0f755c6418478ec3cabb4062a601e/jiant/modules/simple_modules.py#L49
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.classifier = Sequential(
            Linear(input_dim, hidden_dim),
            Tanh(),
            LayerNorm(hidden_dim),
            Dropout(dropout),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, inputs):
        return self.classifier(inputs)
