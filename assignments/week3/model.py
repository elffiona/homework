import torch
from typing import Callable
import numpy as np


class MLP(torch.nn.Module):
    """
    Multilayer perceptron with same size for each layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        # Add input layer
        self.layers.append(torch.nn.Linear(input_size, hidden_size))

        # Add hidden layers
        for i in np.arange(hidden_count):
            self.layers.append(activation())
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))

        # Add output layers
        self.layers.append(activation())
        self.layers.append(torch.nn.Linear(hidden_size, num_classes))

        # Initialize weights
        for lay in self.layers:
            if type(lay) == torch.nn.Linear:
                # initializer(lay.weight)
                initializer(lay.weight, gain=1.4141428569978354)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # Loop through each layer and apply
        for ll in self.layers:
            x = ll(x)

        return x
