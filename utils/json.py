import json
from typing import Union


def int_hook(src):
    if not isinstance(src, dict):
        return
    rv = dict()
    for k, v in src.items():
        try:
            k_ = int(k)
            rv[k_] = v
        except ValueError:
            rv[k] = v
    return rv


def load(fn: str, convert_int: bool = False) -> Union[list, dict]:
    """
    provide a filename and return dict or list
    """
    if convert_int:
        return json.load(open(fn), object_hook=int_hook)
    else:
        return json.load(open(fn))


def dump(obj: Union[list, dict], fn: str, indent=None):
    json.dump(obj, open(fn, 'w'), ensure_ascii=False, indent=indent)
