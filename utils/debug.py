import torch
import pysnooper
import numpy


def large(l):
    return isinstance(l, list) and len(l) > 5


def tensor(t):
    return isinstance(t, torch.Tensor)


def print_list_size(l):
    return 'list(size={})'.format(len(l))


def print_ndarray(a):
    return 'ndarray(shape={}, dtype={})'.format(a.shape, a.dtype)


def debug_tensor(t: torch.Tensor):
    return f"[Shape] {t.shape}"


@pysnooper.snoop(custom_repr=((large, print_list_size), (numpy.ndarray, print_ndarray)))
def sum_to_x(x):
    l = list(range(x))
    a = numpy.zeros((10, 10))
    return sum(l)


@pysnooper.snoop(custom_repr=(torch.Tensor, debug_tensor))
def sum_tensor(x, y):
    a = torch.randn((100, 2))
    return x + y
