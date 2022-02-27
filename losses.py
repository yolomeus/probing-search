from torch.nn import BCEWithLogitsLoss, Module


class BCEWithLongLoss(Module):
    """BCEWithLogitsLoss but target input are being are cast to floats, to allow labels of type long.
    """

    def __init__(self):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets.float())
