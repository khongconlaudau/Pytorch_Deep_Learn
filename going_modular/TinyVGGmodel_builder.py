
"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn
"""Creates the TinyVGG architecture.

Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
See the original architecture here: https://poloclub.github.io/cnn-explainer/

Args:
input_shape: An integer indicating number of input channels.
hidden_units: An integer indicating number of hidden units between layers.
output_shape: An integer indicating number of output units.
"""
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            # output_size = (input_size - kernel_size + 2*padding) / stride + 1
            nn.Conv2d(in_channels=input_shape,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1,
                     padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1,
                     padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13,
                     out_features=output_shape)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)

        return x
