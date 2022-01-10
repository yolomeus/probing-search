from torch.nn import Module, Embedding


class RandomEmbeddings(Module):
    """Randomly initialized Embedding layer.
    """

    def __init__(self, num_embeddings, h_dim, model_name=None):
        """

        :param num_embeddings: Number od different embeddings in the embedding matrix, i.e. vocab size.
        :param h_dim: size of each embedding.
        :param model_name: name of this model.
        """
        super().__init__()

        self.model_name = model_name
        self.embed = Embedding(num_embeddings, h_dim)

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        x = self.embed(input_ids)
        return [x]
