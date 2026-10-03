"""
EEG-Conformer with optional feature output for session-level CORAL.

Input:
    (batch_size, 8, 1200)

Output:
    (batch_size, 5)
"""

import torch
from torch import nn

from braindecode.models import EEGConformer


class EEGConformerWithFeatures(nn.Module):
    """Preserve Braindecode logits while optionally exposing 32-D features."""

    def __init__(
        self,
        n_filters_time=32,
        att_depth=2,
        att_heads=4,
        drop_prob=0.5,
    ):
        super().__init__()

        self.backbone = EEGConformer(
            n_chans=8,
            n_outputs=5,
            n_times=1200,
            n_filters_time=n_filters_time,
            filter_time_length=25,
            pool_time_length=75,
            pool_time_stride=15,
            drop_prob=drop_prob,
            
            num_layers=att_depth,
            num_heads=att_heads,
            att_drop_prob=drop_prob,
            final_fc_length="auto",
            return_features=False,
        )

        self._latest_features = None
        self._feature_hook = self.backbone.fc.register_forward_hook(
            self._capture_features
        )

    def _capture_features(self, module, inputs, output):
        """Capture the official fully connected representation."""
        self._latest_features = output

    def forward(self, x, return_features=False):
        self._latest_features = None
        logits = self.backbone(x)

        if self._latest_features is None:
            raise RuntimeError(
                "EEG-Conformer features were not captured."
            )

        if return_features:
            return logits, self._latest_features

        return logits

def create_eeg_conformer(
    n_filters_time=32,
    att_depth=2,
    att_heads=4,
    drop_prob=0.5,
):
    if n_filters_time % att_heads != 0:
        raise ValueError(
            "n_filters_time must be divisible by att_heads."
        )

    return EEGConformerWithFeatures(
        n_filters_time=n_filters_time,
        att_depth=att_depth,
        att_heads=att_heads,
        drop_prob=drop_prob,
    )



def test_model():
    """Test the model using an artificial batch."""

    model = create_eeg_conformer()

    dummy_eeg = torch.randn(
        4,
        8,
        1200,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():
        output, features = model(
            dummy_eeg,
            return_features=True
        )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Input shape:", dummy_eeg.shape)
    print("Output shape:", output.shape)
    print("Feature shape:", features.shape)
    print("Parameters:", number_of_parameters)

    assert output.shape == (4, 5)
    assert features.shape == (4, 32)

    print("EEG-Conformer test passed.")


if __name__ == "__main__":
    test_model()
