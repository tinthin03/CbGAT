#from torch._six import container_abcs
import torch
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


def cpu(obj, *args, **kwargs):
    """
    Transfer any nested structure to CPU
    """
    if hasattr(obj, "cpu"):
        return obj.cpu(*args, **kwargs)
    elif isinstance(obj, container_abcs.Mapping):
        return {k: cpu(v, *args, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, container_abcs.Sequence):
        return type(obj)(cpu(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested structure to CUDA
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, container_abcs.Mapping):
        return {k: cuda(v, *args, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, container_abcs.Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))