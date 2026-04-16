import torch
from torch.library import custom_op

@custom_op("dcl::watch", mutates_args=())
def watch(tensor: torch.Tensor, name: str = "") -> torch.Tensor:
    """
    Watch a tensor during compiled execution.
    In eager mode, this is a no-op that returns the tensor.
    In compiled mode, DCL will intercept this and record the tensor value.
    """
    return tensor

@watch.register_fake
def _(tensor, name=""):
    return torch.empty_like(tensor)
