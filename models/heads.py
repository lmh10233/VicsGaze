from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

import utils


class ProjectionHead(nn.Module):

    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class BYOLProjectionHead(ProjectionHead):
    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(BYOLProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class BYOLPredictionHead(ProjectionHead):
    def __init__(
        self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(BYOLPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SimSiamProjectionHead(ProjectionHead):
    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 2048
    ):
        super(SimSiamProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (
                    hidden_dim,
                    output_dim,
                    nn.BatchNorm1d(output_dim, affine=False),
                    None,
                ),
            ]
        )


class SimSiamPredictionHead(ProjectionHead):
    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 2048
    ):
        super(SimSiamPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class VICRegProjectionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 8192,
        output_dim: int = 8192,
        num_layers: int = 3,
    ):
        hidden_layers = [
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU())
            for _ in range(num_layers - 2)  # Exclude first and last layer.
        ]
        super(VICRegProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                # (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                *hidden_layers,
                (hidden_dim, output_dim, None, None),
            ]
        )
