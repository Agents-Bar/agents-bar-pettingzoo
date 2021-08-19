import gym
import numpy

from typing import Any, Dict, List

def to_list(obj: object) -> List:
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, numpy.ndarray):
        return obj.tolist()

    # Just try...
    return list(obj)


def consolidate_value(arr: numpy.ndarray):
    flat_arr = arr.flatten()
    if numpy.all(arr == flat_arr[0]):
        return flat_arr[0].item()
    
    return to_list(arr)


def extract_space_info(space) -> Dict[str, Any]:
    if isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
        return dict(
            dtype=str(space.dtype),
            shape=[len(space.nvec)],
            low=0,
            high=to_list(space.nvec),
        )
    elif "Discret" in str(space):
        return dict(
            dtype=str(space.dtype),
            shape=[1],
            low=0,
            high=space.n - 1,  # Inclusive bounds, so n=2 -> [0,1]
        )
    else:
        return dict(
            low = consolidate_value(space.low),
            high = consolidate_value(space.high),
            shape = space.shape,
            dtype = str(space.dtype),
        )
