import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    """
    Input shape: (batch, channels, time_samples)
    Output shape: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        input_time_samples: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Convert (B, C, T) -> (B, 1, C, T)
            nn.Unflatten(1, (1, n_channels)),

            # Block 1: temporal then spatial convolution
            nn.Conv2d(
                1, 25,
                kernel_size=(1, 10),
                padding=(0, 4),
                bias=False,
            ),
            nn.Conv2d(
                25, 25,
                kernel_size=(n_channels, 1),
                bias=False,
            ),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv2d(
                25, 50,
                kernel_size=(1, 10),
                padding=(0, 4),
                bias=False,
            ),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv2d(
                50, 100,
                kernel_size=(1, 10),
                padding=(0, 4),
                bias=False,
            ),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropout),

            # Block 4
            nn.Conv2d(
                100, 200,
                kernel_size=(1, 10),
                padding=(0, 4),
                bias=False,
            ),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropout),
        )

        # Automatically calculate the flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, input_time_samples)
            feature_size = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Linear(feature_size, n_classes)

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        if x.ndim != 3:
            raise ValueError(
                f"DeepConvNet expected 3D input, but received {x.shape}"
        )

        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)  # raw logits