import jax
import jax.numpy as jnp
import mujoco

from flax import struct
from typing import Any, Dict, Optional

from builderbench.env_utils import State

# Adapted from https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py

class Wrapper():
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Any):
        self.env = env

    def reset(self, rng) -> State:
        return self.env.reset(rng)
    
    def post_step(self, state, physics_state, sensor_data):
        return self.env.post_step(state, physics_state, sensor_data)
        
    def pre_step(self, state, action):
        return self.env.pre_step(state, action)
        
    def step(self, state: State, action: jax.Array) -> State:
        return self.env.step(state, action)

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size
    
    @property
    def goal_size(self) -> int:
        return self.env.goal_size

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.env.mj_model

    @property
    def xml_path(self) -> str:
        return self.env.xml_path
    
    
class AutoResetWrapper(Wrapper):
    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_physics_state'] = state.physics_state
        state.info['first_sensordata'] = state.sensordata
        state.info['first_obs'] = state.obs
        state.info['first_ctrl'] = state.ctrl
        state.info['first_achieved_goal'] = state.info['achieved_goal']
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1])

        return state
    
    def pre_step(self, state: State, action: jax.Array) -> State:
        state = self.env.pre_step(state, action)
        
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        return state
  
    def post_step(self, state, physics_state, sensor_data) -> State:

        state = self.env.post_step(state, physics_state, sensor_data)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        physics_state = jax.tree.map(where_done, state.info['first_physics_state'], state.physics_state)
        sensordata = jax.tree.map(where_done, state.info['first_sensordata'], state.sensordata)
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        ctrl = jax.tree.map(where_done, state.info['first_ctrl'], state.ctrl)
        state.info['achieved_goal'] = jax.tree.map(where_done, state.info['first_achieved_goal'], state.info['achieved_goal'])
        state.info['traj_id'] = jnp.where(state.done, state.info['traj_id'] + 1, state.info['traj_id'])
        return state.replace(physics_state=physics_state, sensordata=sensordata, obs=obs, ctrl=ctrl, info=state.info)

class VmapWrapper(Wrapper):
    def __init__(self, env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)
    
    def post_step(self, state, physics_state, sensor_data):
        return jax.vmap(self.env.post_step)(state, physics_state, sensor_data)
        
    def pre_step(self, state, action):
        return jax.vmap(self.env.pre_step)(state, action)

class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        assert action_repeat == 1
        self.action_repeat = action_repeat  # unsued right now

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state

    def post_step(self, state, output_physics_state, output_sensor_data):
        state = self.env.post_step(state, output_physics_state, output_sensor_data)
        
        steps = state.info['steps'] + 1

        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps
        return state.replace(done=done)

@struct.dataclass
class EvalMetrics:
    episode_metrics: Dict[str, jax.Array]
    active_episodes: jax.Array
    episode_steps: jax.Array

class EvalWrapper(Wrapper):
    def reset(self, rng: jax.Array):
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward

        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(
                jnp.zeros_like, reset_state.metrics
            ),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state
        
    def pre_step(self, state, action):
        return self.env.pre_step(state, action)

    def step(self, state, action):
        return self.env.step(state, action)

    def post_step(self, state, physics_state, sensor_data):
        state_metrics = state.info['eval_metrics']
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(
                f'Incorrect type for state_metrics: {type(state_metrics)}'
            )
        del state.info['eval_metrics']

        state = self.env.post_step(state, physics_state, sensor_data)

        state.metrics['reward'] = state.reward

        episode_steps = jnp.where(
            state_metrics.active_episodes,
            state.info['steps'],
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            state.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - state.done)

        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        state.info['eval_metrics'] = eval_metrics

        return state

def wrap_env(
    env,
    episode_length: int = 150,
    action_repeat: int = 1,
) -> Wrapper:

    env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = AutoResetWrapper(env)
    return env