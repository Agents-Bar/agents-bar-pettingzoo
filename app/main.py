import importlib
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.params import Depends

from .types import ActionType, EnvActionType, EnvInfo, EnvStepType, ObservationType
from .utils import extract_space_info, to_list

app = FastAPI()

global_env = None
last_action: Optional[ActionType] = None

def get_env():
    global global_env
    if global_env is None:
        raise HTTPException(400, "Environment hasn't been initiated. Call POST `/env` to initiate.")
    return global_env


@app.get("/ping")
def ping():
    return "pong"


@app.post("/env", status_code=201)
def api_post_env_create(config: Dict[str, Any]):
    "Create environment based on the OpenAI Gym"
    global global_env
    env_name: Optional[str] = config.get('env_name')
    if env_name is None:
        raise HTTPException(400, "Requires passing 'env_name' key to setup environment")
    elif ":" not in env_name:
        raise HTTPException(400, "Expected 'env_name' to contain '{core}:{gym_name}', e.g. 'butterfly:prison_v3'")
    
    core, gym_name = env_name.split(":")
    env_maker = importlib.import_module(f"pettingzoo.{core}.{gym_name}")

    global_env = env_maker.env()
    if global_env is None:
        raise HTTPException(500, "Failed to create environment")
    return str(global_env)


@app.post('/env/reset', response_model=ObservationType)
def api_post_env_reset(seed: Optional[int] = None, env = Depends(get_env)) -> ObservationType:
    "Reset the environment to initial position."
    return env.reset()


@app.post("/env/step", response_model=Optional[EnvStepType])
def api_post_env_step(env_action: EnvActionType, env = Depends(get_env)):
    "Provides information necessary to step the environment."
    global last_action
    if env_action.actor_name != env.agent_selection:
        raise HTTPException(400, f"Expected actor to be '{env.agent_selection}' but passed '{env_action.actor_name}'")

    last_action = env_action.actions
    if env_action.commit:
        return env_commit_action()
    return None


@app.post('/env/commit', response_model=Optional[EnvStepType])
def api_post_env_commit(env = Depends(get_env)) -> Optional[EnvStepType]:
    "Commit last provided Step data"
    if last_action is None:
        raise HTTPException(400, "No action available. Make sure to call `/env/step` before attempting submission.")
    return env_commit_action()


@app.get('/env/last', response_model=EnvStepType)
def get_last(env = Depends(get_env)) -> EnvStepType:
    "Retrieves last provided Step data"
    last_step = env.last()
    obs = to_list(last_step[0])
    reward = last_step[1]
    done = last_step[2]
    info = last_step[3]
    actor_name = env.agent_selection
    return EnvStepType(observation=obs, reward=reward, done=done, info=info, actor_name=actor_name)


@app.post('/env/seed')
def api_post_env_seed_set(seed: int, env = Depends(get_env)) -> None:
    "Set seed for environment's random number generator."
    env.seed(seed)
    return None

    
@app.get('/env/info', response_model=EnvInfo)
def api_get_env_info(env = Depends(get_env)) -> EnvInfo:
    """Environment related information.
    
    The very least this method should provide:
    * key: observation_space, value: expected Space type for observation
    * key: action_space, value: expected Space type for action
    * key: num_agents, value: int of agents to support

    """

    info = EnvInfo(
        observation_space={k: extract_space_info(v) for (k, v) in env.observation_spaces.items()},
        action_space={k: extract_space_info(v) for (k, v) in env.action_spaces.items()},
        num_agents=env.num_agents,
        actor_names=env.agents,
        reward_range=None
    )
    return info


def env_commit_action() -> Optional[EnvStepType]:
    global globa_env, last_action
    step = global_env.step(last_action)
    if step is not None:
        obs = to_list(step[0])
        reward = step[1]
        done = step[2]
        info = step[3]
        return EnvStepType(observation=obs, reward=reward, done=done, info=info)
    return None
