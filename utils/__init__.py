from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import typer

runner = typer.Typer(pretty_exceptions_enable=False)


def time_suffix(last: bool = False):
    if last:
        return str(datetime.now()).split(' ')[-1].replace(':', '').replace('.', '_')
    return str(datetime.now()).split(' ')[0]


class DictMixIn:

    @classmethod
    def from_dict(cls, kwargs):
        obj = cls()
        for k, v in kwargs.items():
            cls.__setattr__(obj, k, v)

        return obj

    def to_dict(self, *skip_keys):
        result = dict(self.__class__.__dict__)
        o = dict()

        for k, v in result.items():
            if str(k).startswith('_') or k in skip_keys:
                pass
            else:
                if isinstance(v, DictMixIn):
                    o[k] = v.to_dict('from_dict', 'to_dict')
                else:
                    o[k] = v
        obj_dict = self.__dict__
        for k, v in obj_dict.items():
            if str(k).startswith('_') or k in skip_keys:
                pass
            else:
                if isinstance(v, DictMixIn):
                    o[k] = v.to_dict('from_dict', 'to_dict')
                else:
                    o[k] = v
        return dict(sorted(o.items()))


def partition(ls: list, size: int):
    return [ls[i:i + size] for i in range(0, len(ls), size)]
