from torch.nn import Linear, ReLU, Sequential, Dropout, Module


class MLP(Module):
    """Simple Multi-Layer Perceptron also known as Feed-Forward Neural Network."""

    def __init__(self,
                 in_dim: int,
                 h_dim: int,
                 out_dim: int,
                 dropout: float):
        """

        :param in_dim: input dimension
        :param h_dim: hidden dimension
        :param out_dim: output dimension
        :param dropout: dropout rate
        """

        super().__init__()
        self.classifier = Sequential(Dropout(dropout),
                                     Linear(in_dim, h_dim),
                                     ReLU(),
                                     Dropout(dropout),
                                     Linear(h_dim, out_dim))

    def forward(self, inputs):
        x = self.classifier(inputs)
        return x
