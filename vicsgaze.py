import warnings

import torch
import torch.nn as nn

from models.heads import VICRegProjectionHead


class VicsGaze(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        num_ftrs: int = 512,
        hidden_dim: int = 1024,
        out_dim: int = 512,
        num_layer: int = 3
    ):
        super(VicsGaze, self).__init__()

        self.backbone = backbone
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.projection_mlp = VICRegProjectionHead(
            num_ftrs,
            hidden_dim,
            out_dim,
            num_layer
        )

        warnings.warn(
            Warning(
                "The high-level building block SimSiam will be deprecated in version 1.3.0. "
                + "Use low-level building blocks instead. "
                + "See https://docs.lightly.ai/self-supervised-learning/lightly.models.html for more information"
            ),
            DeprecationWarning,
        )

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor = None, return_features: bool = False
    ):
        f0 = self.backbone(x0).flatten(start_dim=1) #　256×512 B×C
        z0 = self.projection_mlp(f0)
        out0 = z0

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0

        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_mlp(f1)
        out1 = z1

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        return out0, out1
    

if __name__ == "__main__":
    import online_resnet
    resnet = online_resnet.create_Res()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = VicsGaze(backbone).cuda()
    x1 = torch.randn(256, 3, 224, 224).cuda()

    print(model(x1))
