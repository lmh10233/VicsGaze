import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from vic_loss.dist import gather
from torch.nn.functional import cosine_similarity


class VicsLoss(torch.nn.Module):

    def __init__(
        self,
        lambda_param: float = 1.0,
        mu_param: float = 1.0,
        nu_param: float = 0.04,
        gama_param: float = 0.8,
        gather_distributed: bool = False,
        eps=0.0001,
    ):
        super(VICRegLoss, self).__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gama_param = gama_param
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert (
            z_a.shape == z_b.shape
        ), f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        # invariance term of the loss
        inv_loss = invariance_loss(x=z_a, y=z_b)

        # gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        var_loss = 0.5 * (
            variance_loss(x=z_a, eps=self.eps) + variance_loss(x=z_b, eps=self.eps)
        )
        cov_loss = covariance_loss(x=z_a) + covariance_loss(x=z_b)

        similar_loss = negativeCosineSimilarity(z_a, z_b)
        
        loss = self.gama_param * (1 + similar_loss) # + self.lambda_param * inv_loss + self.nu_param * cov_loss + self.mu_param * var_loss + self.gama_param * (1 + similar_loss)
        # 不变性，差异性，相似性 self.nu_param * similar_loss
        # print(self.lambda_param * inv_loss, self.mu_param * var_loss, self.nu_param * cov_loss, self.gama_param * (1 + similar_loss))
        print(self.gama_param * (1 + similar_loss))
        return loss


def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
    return F.mse_loss(x, y)


def variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))
    return loss


def covariance_loss(x: Tensor) -> Tensor:
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)
    # cov has shape (..., dim, dim)
    cov = torch.einsum("b...c,b...d->...cd", x, x) / (batch_size - 1)
    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()


def negativeCosineSimilarity(x0: Tensor, x1:Tensor) -> Tensor:
    # def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
    dim = 1
    eps = 1e-8
    return -cosine_similarity(x0, x1, dim, eps).mean()
