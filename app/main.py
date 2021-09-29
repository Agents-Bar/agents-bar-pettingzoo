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
def api_post_env_create(gym_name: str, config: Optional[Dict[str, Any]]=None):
    "Create environment based on the OpenAI Gym"
    global global_env
    # gym_name: Optional[str] = config.get('gym_name')
    if gym_name is None:
        raise HTTPException(400, "Requires passing 'gym_name' key to setup environment")
    elif ":" not in gym_name:
        raise HTTPException(400, "Expected 'gym_name' to contain '{core}:{gym_name}', e.g. 'butterfly:prison_v3'")
    
    core, gym_name = gym_name.split(":")
    env_maker = importlib.import_module(f"pettingzoo.{core}.{gym_name}")

    global_env = env_maker.env()
    if global_env is None:
        raise HTTPException(500, "Failed to create environment")
    
    # PettingZoo requires reset after initialization so that config is populated
    global_env.reset()

    return str(global_env)


@app.post('/env/reset', response_model=ObservationType)
def api_post_env_reset(seed: Optional[int] = None, env = Depends(get_env)) -> ObservationType:
    "Reset the environment to initial position."
    return env.reset()


@app.post("/env/step", response_model=Optional[EnvStepType])
def api_post_env_step(env_action: EnvActionType, env = Depends(get_env)):
    "Provides information necessary to step the environment."
    global last_action
    if env_action.agent_name != env.agent_selection:
        raise HTTPException(400, f"Expected actor to be '{env.agent_selection}' but passed '{env_action.agent_name}'")

    last_action = env_action.actions
    if env_action.commit:
        try:
            return env_commit_action()
        except Exception as e:
            raise HTTPException(400, detail="Couldn't step environment. Reason:\n" + str(e))
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
    agent_name = env.agent_selection
    return EnvStepType(observation=obs, reward=reward, done=done, info=info, agent_name=agent_name)


@app.get('/env/agent', response_model=str)
def get_env_agent_current(env = Depends(get_env)) -> str:
    """Returns name of currently expected agent."""
    return env.agent_selection


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
        agent_names=env.agents,
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
