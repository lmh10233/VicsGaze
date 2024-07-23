from typing import Optional, Tuple

import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ctx.save_for_backward(input)
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        grad_out = torch.empty_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def rank() -> int:
    """Returns the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Returns the current world size (number of distributed processes)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def gather(input: torch.Tensor) -> Tuple[torch.Tensor]:
    """Gathers this tensor from all processes. Supports backprop."""
    return GatherLayer.apply(input)


def eye_rank(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool)
    diag_mask[(rows, cols)] = True
    return diag_mask


def rank_zero_only(fn):

    def wrapped(*args, **kwargs):
        if rank() == 0:
            return fn(*args, **kwargs)

    return wrapped


@rank_zero_only
def print_rank_zero(*args, **kwargs) -> None:
    """Equivalent to print, but only runs on the process with rank 0."""
    print(*args, **kwargs)
