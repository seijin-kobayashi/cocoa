from .base import Environment
from .interleaving import InterleavingStochastic
from .treasure_conveyor import ConveyorTreasure
from .tree import Tree


_envs = {
    "interleaving_stochastic": InterleavingStochastic,
    "treasure_conveyor": ConveyorTreasure,
    "tree": Tree,
}


def create(name: str, **kwargs) -> Environment:
    """Creates an Env with a specified brax system."""
    env = _envs[name](**kwargs)
    return env
