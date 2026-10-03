"""Official Braindecode EEGNet-v4 model for ArEEG."""

from __future__ import annotations

import torch
from braindecode.models import EEGNet


def create_eegnet(
    n_channels: int = 8,
    n_classes: int = 5,
    n_times: int = 1200,
    drop_prob: float = 0.5,
) -> EEGNet:
    """Create EEGNet-v4 for 8-channel, five-class ArEEG trials."""
    return EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length="auto",
        drop_prob=drop_prob,
    )


def test_model() -> None:
    """Run a small forward-pass check before full training."""
    model = create_eegnet()
    inputs = torch.randn(4, 8, 1200)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

    if outputs.shape != (4, 5):
        raise RuntimeError(
            f"Expected output shape (4, 5), received {tuple(outputs.shape)}."
        )

    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)
    print(
        "Parameters:",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    print("EEGNet-v4 test passed.")


if __name__ == "__main__":
    test_model()
