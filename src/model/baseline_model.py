from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, n_feats: int, n_class: int, fc_hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feats, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_class),
        )

    def forward(self, **batch):
        raise NotImplementedError(
            "Run inputs through self.net and return a dict with at least a 'loss' key "
            "(trainer calls batch['loss'].backward())."
        )

    def __str__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + f"\nTotal params: {total}\nTrainable params: {trainable}"
