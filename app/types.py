from __future__ import annotations  # Python3.8+ support

from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, StrictFloat, StrictInt


# TODO: This implementation takes a long time to parse and has plenty of "__root__=" in parse 
# class NumericalList(BaseModel):
#     __root__: Union[StrictInt, StrictFloat, List[NumericalList]]

# NumericalList.update_forward_refs()  # Required to rollout recurance
NumericalList = Union[
    StrictFloat, List[StrictFloat], List[List[StrictFloat]], List[List[List[StrictFloat]]],
    StrictInt, List[StrictInt], List[List[StrictInt]], List[List[List[StrictInt]]],
]


ActionType = NumericalList
ObservationType = NumericalList


class EnvStepType(BaseModel):
    agent_name: str
    observation: ObservationType
    reward: float
    done: bool
    info: Dict


class EnvActionType(BaseModel):
    agent_name: str
    actions: ActionType
    commit: bool = False


class EnvInfo(BaseModel):
    observation_space: Dict
    action_space: Dict
    num_agents: int
    agent_names: List[str]
    reward_range: Optional[Tuple[float, float]]
