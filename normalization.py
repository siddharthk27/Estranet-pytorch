import torch
import torch.nn as nn


class LayerScaling(nn.Module):
    """Layer scaling normalization (not used in final model but included for completeness)."""

    def __init__(self, normalized_shape, epsilon=0.001):
        super(LayerScaling, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, inputs):
        """
        Args:
            inputs: tensor of shape [..., normalized_shape]
        """
        variance = inputs.var(dim=-1, keepdim=True, unbiased=False)
        outputs = inputs / torch.sqrt(variance + self.epsilon) * self.gamma
        return outputs


class LayerCentering(nn.Module):
    """Layer centering normalization - centers activations around learned offset."""

    def __init__(self, normalized_shape):
        super(LayerCentering, self).__init__()
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, inputs):
        """
        Args:
            inputs: tensor of shape [..., normalized_shape]
        """
        mean = inputs.mean(dim=-1, keepdim=True)
        outputs = inputs - mean + self.beta
        return outputs
