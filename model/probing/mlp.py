from abc import ABC, abstractmethod

from torch.nn import Module, Sequential, Linear, Tanh, LayerNorm, Dropout, ReLU


class MLP(Module, ABC):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, single_span):
        super().__init__()

        if not single_span:
            input_dim *= 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.classifier = self._build_mlp()

    def forward(self, inputs):
        return self.classifier(inputs)

    @abstractmethod
    def _build_mlp(self):
        """build the mlp classifier

        :rtype: Module
        :return: the mlp module.
        """


class TenneyMLP(MLP):
    """The 2 layer MLP used by Tenney et al. in https://arxiv.org/abs/1905.06316.

    https://github.com/nyu-mll/jiant/blob/ead63af002e0f755c6418478ec3cabb4062a601e/jiant/modules/simple_modules.py#L49
    """

    def _build_mlp(self):
        return Sequential(
            Linear(self.input_dim, self.hidden_dim),
            Tanh(),
            LayerNorm(self.hidden_dim),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.output_dim)
        )


class HewittMLP(MLP):
    """MLP-2 from Hewitt and Liang: https://arxiv.org/abs/1909.03368.
    """

    def _build_mlp(self):
        return Sequential(
            Linear(self.input_dim, self.hidden_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.hidden_dim),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_dim, self.output_dim)
        )
