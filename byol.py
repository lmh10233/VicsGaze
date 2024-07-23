
import warnings
import copy
import torch
import torch.nn as nn
from utils import deactivate_requires_grad, update_momentum
from _momentum import _MomentumEncoderMixin
from models.heads import BYOLPredictionHead, BYOLProjectionHead


def _get_byol_mlp(num_ftrs: int, hidden_dim: int, out_dim: int):
    modules = [
        nn.Linear(num_ftrs, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    ]
    return nn.Sequential(*modules)


class BYOL(nn.Module, _MomentumEncoderMixin):

    def __init__(
        self,
        backbone: nn.Module,
        num_ftrs: int = 512,
        hidden_dim: int = 1024,
        out_dim: int = 512,
        m: float = 0.9,
    ):
        super(BYOL, self).__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(num_ftrs, hidden_dim, out_dim)
        self.prediction_head = BYOLProjectionHead(out_dim, hidden_dim, out_dim)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        
        deactivate_requires_grad(self.backbone_momentum)  # 不求梯度
        deactivate_requires_grad(self.projection_head_momentum)

        self._init_momentum_encoder()
        self.m = m

        warnings.warn(
            Warning(
                "The high-level building block BYOL will be deprecated in version 1.3.0. "
                + "Use low-level building blocks instead. "
                + "See https://docs.lightly.ai/self-supervised-learning/lightly.models.html for more information"
            ),
            DeprecationWarning,
        )

    def _forward(self, x0: torch.Tensor, x1: torch.Tensor = None):

        self._momentum_update(self.m)

        # forward pass of first input x0
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)
        out0 = self.prediction_head(z0)
        
        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.no_grad():
            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            out1 = self.momentum_projection_head(f1)

        return out0, out1

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, return_features: bool = False
    ):

        if x0 is None:
            raise ValueError("x0 must not be None!")
        if x1 is None:
            raise ValueError("x1 must not be None!")

        if not all([s0 == s1 for s0, s1 in zip(x0.shape, x1.shape)]):
            raise ValueError(
                f"x0 and x1 must have same shape but got shapes {x0.shape} and {x1.shape}!"
            )

        p0, z1 = self._forward(x0, x1)
        p1, z0 = self._forward(x1, x0)

        return (z0, p0), (z1, p1)