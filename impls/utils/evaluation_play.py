import time
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import rollout

from typing import Callable

from utils.wrapper import EvalWrapper

def get_trajectory(policy, env, key, unroll_length):
    @jax.jit
    def _get_trajectory(key):
        env_key, key = jax.random.split(key)
        eval_first_state = env.reset(jax.random.split(env_key, 1))
        
        def f(carry, unused_t):
            env_state, key = carry
            key, next_key = jax.random.split(key)
            action, _ = policy(env_state.obs, env_state.info["target_goal"], env_state.info["goal_mask"], key) 

            next_env_state = env.pre_step(env_state, action)  
            physics_state, sensor_data = env.step(next_env_state, action)
            next_env_state = env.post_step(next_env_state, physics_state, sensor_data)

            return (next_env_state, next_key), next_env_state
        
        _, env_states = jax.lax.scan(
                f, 
                (eval_first_state, key), 
                (), 
                length=unroll_length
            )
        return env_states
    
    return _get_trajectory(key)

def generate_unroll(env, env_state, policy, key, unroll_length):
    """Step the env for unroll_length steps and return final state."""
    def body(i, carry):
        env_state, key = carry
        key, next_key = jax.random.split(key)
        actions, _ = policy(env_state.obs, env_state.info["target_goal"], env_state.info["goal_mask"], key)
        next_env_state = env.pre_step(env_state, actions)  
        physics_state, sensor_data = env.step(next_env_state, actions)
        next_env_state = env.post_step(next_env_state, physics_state, sensor_data)
        return (next_env_state, next_key)

    final_state, _ = jax.lax.fori_loop(
        0, unroll_length, body, (env_state, key)
    )
    return final_state

class Evaluator:
    def __init__(
        self,
        eval_env,
        eval_policy_fn: Callable,
        num_eval_envs: int,
        episode_length: int,
        key: jax.Array,
    ):
    
        self._key = key
        self._eval_walltime = 0.0

        eval_env = EvalWrapper(eval_env)

        def generate_eval_unroll(policy_params, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                key,
                unroll_length=episode_length,
            )

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params,
        training_metrics,
    ):
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)

        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}

        unique_task_ids = jnp.unique( eval_state.info['task_id'] ).reshape(-1, 1)
        num_task_ids = len(unique_task_ids)
        task_based_masks = eval_state.info['task_id'] == unique_task_ids

        for name, value in eval_metrics.episode_metrics.items():

            if name in ["success", "easy_success", "hard_success", "reward", "obj_lifted", "obj_moved"]:
                value_ = jnp.repeat( value[jnp.newaxis], repeats=num_task_ids, axis=0 )
                value_means = jnp.sum(value_ * task_based_masks, axis=-1) / jnp.sum( task_based_masks, axis=-1)

                if name in ["success", "easy_success", "hard_success"]:
                    value_rate_means = jnp.sum(value_ * task_based_masks > 0.0, axis=-1) / jnp.sum( task_based_masks, axis=-1)

                for i in range(num_task_ids):
                    metrics.update({
                        f'eval/task_{unique_task_ids[i, 0] + 1}__episode_{name}': (
                            value_means[i]
                        )
                    })

                    if name in ["success", "easy_success", "hard_success"]:
                        metrics[f'eval/task_{unique_task_ids[i, 0] + 1}__episode_{name}_rate'] = value_rate_means[i]

        metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
        metrics['eval/epoch_eval_time'] = epoch_eval_time
        metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            'eval/walltime': self._eval_walltime,
            **training_metrics,
            **metrics,
        }

        return metrics
    
def get_video(env_id, inference_policy, video_env, video_key, episode_length):

    camera = mujoco.MjvCamera()
    camera.distance = 0.8
    camera.lookat = np.array([0.4, 0.0 , 0.4])
    camera.elevation = -30.0
    camera.azimuth = 180


    rollout.shutdown_persistent_pool()
    video_env_states = get_trajectory(inference_policy, video_env, video_key, episode_length)
    rollout.shutdown_persistent_pool()

    if 'goal_mask' in video_env_states.info:
        video_env.model.geom_rgba[ video_env._mocap_targets_geom, -1 ] = 0.2
        video_env.model.geom_rgba[ video_env._mocap_targets_geom[ ~video_env_states.info['goal_mask'][0][0] ], -1] = 0

    mocap_key = 'target_mocap'
    
    video_images = []
    for i in range(episode_length):
        if i % 2 == 0:
            video_images.append(video_env.render_from_info(
                video_env_states.physics_state[i][0, 1:][:video_env.model.nq], 
                video_env_states.physics_state[i][0, 1:][video_env.model.nq: video_env.model.nq + video_env.model.nv], 
                video_env_states.info[f'{mocap_key}_pos'][i][0],
                video_env_states.info[f'{mocap_key}_quat'][i][0],
            ))
    return video_images