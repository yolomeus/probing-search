from torch.nn import Module


class ProbingPair(Module):
    """A model setup for probing, consisting of a subject model, pooler and a probing model. The pooler acts as
    interface between the subject model and probing model.
    """

    def __init__(self, subject_model: Module, pooler: Module, probing_model: Module, freeze_subject=True):
        super().__init__()
        self.subject_model = subject_model
        self.pooler = pooler
        self.probing_model = probing_model

        if freeze_subject:
            self._freeze_model(subject_model)

    @staticmethod
    def _freeze_model(model: Module):
        """Exclude all parameters of `model` from any gradient updates.

        :param model: freeze all parameters of this model.
        """
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        """
        :param inputs: a tuple of subject model inputs, and target spans. The spans are needed for the pooler to select
        and pool the correct embeddings.
        """

        x, target_spans = inputs
        hidden_states = self.subject_model(x)
        x = self.pooler(hidden_states, target_spans)
        y = self.probing_model(x)
        return y
