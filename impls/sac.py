import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import time
import tyro
import numpy as np
import functools
import pprint
import wandb
import wandb_osh

import jax
import flax
import optax
import distrax
import flax.linen as nn
import jax.numpy as jnp

from mujoco import rollout
from pathlib import Path
from flax.training.train_state import TrainState
from dataclasses import dataclass
from typing import Any, Sequence, NamedTuple
from wandb_osh.hooks import TriggerWandbSyncHook

import utils.running_statistics as running_statistics
from utils.buffer import TrajectoryUniformSamplingQueue
from utils.wrapper import wrap_env
from utils.evaluation import Evaluator
from utils.networks import MLP, save_params
from builderbench.env_utils import make_env

@dataclass
class Args:
    # experiment
    agent: str = "sac"
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    
    # logging and checkpointing
    track: bool = False
    wandb_project_name: str = "builderbench"
    wandb_entity: str = None
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'default'
    wandb_name_tag: str = ''

    num_eval_steps: int = 50             # number of evaluation / logging / saving steps
    num_reset_steps: int = 1             # number of times to call true resets (env.reset) instead of soft resets (AutoResetWrapper)

    save_checkpoint: bool = True

    # environment
    env_id: str = 'cube-1-task1'
    num_envs: int = 2048
    num_eval_envs: int = 128
    num_threads: int = 12
    env_early_termination: bool = True
    permutation_invariant_reward: bool = True   # invariance to the order of cubes in any structure

    # algorithm
    num_timesteps: int = 50000000
    rollout_length: int = 64
    batch_size: int = 4096
    sequence_length: int = 512
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    discount: float = 0.99
    tau: float = 0.01
    entropy_cost: float = 0.1
    max_replay_size: int = 10000
    min_replay_size: int = 1000

class CriticTrainState(TrainState):
    """trainstate for critic that also stores target parameters"""
    target_params: Any
  
@flax.struct.dataclass
class SACTrainingState:
    """Contains training state for the learner"""
    normalizer_params: Any
    env_steps: float
    actor_state: TrainState
    critic_state: TrainState
    
class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    commanded_goal: jnp.ndarray
    achieved_goal: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()

class Critic(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = False
    
    def setup(self):
        self.q1 = MLP(
            layer_sizes=self.layer_sizes, 
            activation=self.activation, 
            layer_norm=self.layer_norm
        )

        self.q2 = MLP(
            layer_sizes=self.layer_sizes, 
            activation=self.activation, 
            layer_norm=self.layer_norm
        )

    def __call__(self, observations, action, goals, normalizer_params=None):
        observation_goals = jnp.concatenate([observations, goals], axis=-1)
        if normalizer_params is not None:
            observation_goals = (observation_goals - normalizer_params.mean ) / (normalizer_params.std)
        inputs = jnp.concatenate([observation_goals, action], axis=-1)
        q1 = self.q1(inputs)
        q2 = self.q2(inputs)
        return jnp.concatenate([q1, q2], axis=-1)
    
class Actor(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = False
    _min_std: float = 0.001
    _var_scale: float = 1
    
    def setup(self):
        self.actor_net = MLP(self.layer_sizes, activation=self.activation, layer_norm=self.layer_norm)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
        stats = self.actor_net(x)
        loc, scale = jnp.split(stats, 2, axis=-1)
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale

        return distrax.Normal(loc=loc, scale=scale)

def make_inference_fn(policy_network):
    """Creates params and inference function for the SAC agent."""
    def make_policy(params, deterministic: bool = False):
        bijector = distrax.Tanh()  

        def policy(observations, goals, key_sample):
            inputs = jnp.concatenate([observations, goals], axis=-1)
            policy_dist = policy_network.apply(params['policy'], inputs, params['normalizer'])
                
            if deterministic:
                return bijector.forward( policy_dist.mode() ), {}
                
            raw_actions = policy_dist.sample(seed=key_sample)
                
            log_prob = policy_dist.log_prob(raw_actions) - bijector.forward_log_det_jacobian(raw_actions)
            log_prob = jnp.sum(log_prob, axis=-1)  
            postprocessed_actions = bijector.forward(
                raw_actions
            )
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions,
            }

        return policy
    return make_policy

def main(args: Args):
    
    args.num_training_step = args.num_timesteps // ( args.num_envs * args.rollout_length )
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps

    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = { (args.sequence_length * args.num_envs) // args.batch_size}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")
    print(f"Data to update ratio = {  ( args.num_envs * args.rollout_length ) / ( args.sequence_length * args.num_envs // args.batch_size )}") 

    args.exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag != '' else ''}{args.env_id}__{args.seed}__{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"

    # Initialize wandb if tracking is enabled
    if args.track:            
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            group=args.wandb_group,
            name=args.exp_name,
            config=vars(args),
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_key, actor_key, critic_key = jax.random.split(key, 6)

    # Initialize environment
    env_class, default_config = make_env(args)
    env = wrap_env( env_class(num_envs=args.num_envs, num_threads=args.num_threads, config=default_config), default_config.episode_length )
    eval_env = wrap_env( env_class(num_envs=args.num_eval_envs, num_threads=args.num_threads, config=default_config), default_config.episode_length )  
    episode_length = default_config.episode_length

    # Initialize checkpoint folder
    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f"checkpoints/{args.exp_name}/"
        os.makedirs(save_path, exist_ok=True)

    reset_fn = jax.jit(env.reset)
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = reset_fn(env_keys)
    obs_size = env.observation_size
    action_size = env.action_size
    goal_size = env.goal_size

    log_data_metric_keys = []
    for k in ("obj_reached_once", "obj_lifted", "obj_moved"):
        if k in env_state.metrics.keys():
            log_data_metric_keys.append(k)
    log_data_metric_keys = tuple(log_data_metric_keys)

    # Network setup
    # Actor
    actor = Actor(layer_sizes=[1024,] * 4 + [action_size * 2])
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, x=jnp.zeros((1, obs_size+goal_size))),
        tx=optax.adam(learning_rate=args.actor_learning_rate)
    )

    # Critic
    critic = Critic(layer_sizes=[1024,] * 4 + [1])
    tmp_init_critic_params = critic.init(critic_key, np.ones([1, obs_size]), np.ones([1, action_size]), np.ones([1, goal_size]))
    critic_state = CriticTrainState.create(
        apply_fn=critic.apply,
        params=tmp_init_critic_params,
        target_params=tmp_init_critic_params,
        tx=optax.adam(learning_rate=args.critic_learning_rate)
    )
    del tmp_init_critic_params

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    make_policy = make_inference_fn(actor)

    # Trainstate
    training_state = SACTrainingState(
        normalizer_params=running_statistics.init_state((obs_size+goal_size,) ),
        env_steps=np.zeros((), dtype=np.float64),
        actor_state=actor_state,
        critic_state=critic_state,
    )

    #Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_goal = jnp.zeros((goal_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        commanded_goal=dummy_goal,
        achieved_goal=dummy_goal,
        action=dummy_action,
        reward=jnp.zeros(()),
        done=jnp.zeros(()),
        next_observation=dummy_obs,
        extras={
            "state_extras": {
                "truncation": 0.0,
            }        
        },
    )
    def jit_wrap(buffer):
        buffer.insert = jax.jit(buffer.insert)
        buffer.sample = jax.jit(buffer.sample)
        return buffer
    
    replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=args.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=args.batch_size,
                num_envs=args.num_envs,
                sequence_length=args.sequence_length,
            )
        )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # Initialize evaluators
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=eval_key,
    )

    def actor_step(training_state, env, env_state, key, extra_fields, metrics_fields):
        policy_inputs = jnp.concatenate([env_state.obs, env_state.info['target_goal']], axis=-1)
        policy_dist = actor.apply(training_state.actor_state.params, policy_inputs, training_state.normalizer_params)
        actions = distrax.Tanh().forward( policy_dist.sample(seed=key) )

        next_env_state = env.pre_step(env_state, actions)  
        physics_state, sensor_data = env.step(next_env_state, actions)
        next_env_state = env.post_step(next_env_state, physics_state, sensor_data)
        
        state_extras = {x: next_env_state.info[x] for x in extra_fields}
        metrics = {x: next_env_state.metrics[x] for x in metrics_fields}

        return training_state, next_env_state, Transition(
                                            observation=env_state.obs,
                                            commanded_goal=env_state.info['target_goal'],
                                            achieved_goal=env_state.info['achieved_goal'],
                                            action=actions,
                                            reward=next_env_state.reward,
                                            done=next_env_state.done,
                                            next_observation=next_env_state.obs,
                                            extras={"state_extras": state_extras},
                                        ), metrics
    
    @jax.jit
    def data_collect_step(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            training_state, env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            training_state, env_state, transition, metrics = actor_step(
                training_state, 
                env, 
                env_state, 
                current_key, 
                extra_fields=("truncation",), 
                metrics_fields=log_data_metric_keys,
            )

            return (training_state, env_state, next_key), (transition, metrics)

        (training_state, env_state, _), (data, metrics) = jax.lax.scan(f, (training_state, env_state, key), (), length=args.rollout_length)

        # Update normalization params.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            jnp.concatenate( [data.observation, data.achieved_goal], axis=-1 )
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + (args.num_envs * args.rollout_length),
        )

        buffer_state = replay_buffer.insert(buffer_state, data)
        
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            training_state, env_state, buffer_state, _ = data_collect_step(
                training_state,
                env_state,
                buffer_state,
                key,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=np.ceil(args.min_replay_size / args.rollout_length))[0]

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        def actor_loss(actor_params, critic_params, transitions, key):

            # prepare reward data
            state = transitions.observation
            commanded_goal = transitions.commanded_goal
        
            actor_dist = actor.apply(
                actor_params, 
                jnp.concatenate([state, commanded_goal], axis=-1),
                training_state.normalizer_params,
            )
            raw_action = actor_dist.sample(seed=key)
            log_prob = actor_dist.log_prob(raw_action) - distrax.Tanh().forward_log_det_jacobian(raw_action)
            log_prob = jnp.sum(log_prob, axis=-1) 
            action = distrax.Tanh().forward(
                raw_action
            ) 

            qf_pi = critic.apply(
                jax.lax.stop_gradient(critic_params), 
                state, 
                action, 
                commanded_goal,
                training_state.normalizer_params
            )
            qf_pi = jnp.min(qf_pi, axis=-1)

            actor_loss = jnp.mean( args.entropy_cost * log_prob - qf_pi )

            return actor_loss, log_prob
        
        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(training_state.actor_state.params, training_state.critic_state.params, transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        training_state = training_state.replace(actor_state=new_actor_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
        }

        return training_state, metrics
    
    @jax.jit
    def update_critic(transitions, training_state, key):
        def critic_loss(critic_params, critic_target_params, actor_params, normalizer_params, transitions, key):
            
            # prepare goal and reward data
            commanded_goal = transitions.commanded_goal
            reward = transitions.reward
            
            # get params
            actor_params = jax.lax.stop_gradient(actor_params)
            target_critic_params = jax.lax.stop_gradient(critic_target_params)
            
            next_actor_dist = actor.apply(
                actor_params, 
                jnp.concatenate([transitions.next_observation, commanded_goal], axis=-1),
                normalizer_params,
            )
            next_raw_action = next_actor_dist.sample(seed=key)
            next_log_prob = next_actor_dist.log_prob(next_raw_action) - distrax.Tanh().forward_log_det_jacobian(next_raw_action)
            next_log_prob = jnp.sum(next_log_prob, axis=-1) 
            next_action = distrax.Tanh().forward(
                next_raw_action
            ) 

            next_v = jnp.min( critic.apply(target_critic_params, transitions.next_observation, next_action, commanded_goal, normalizer_params), axis=-1 ) - args.entropy_cost * next_log_prob
            target_q = reward + args.discount * (1 - transitions.done) * next_v
            q = critic.apply(critic_params, transitions.observation, transitions.action, commanded_goal, normalizer_params)
            q_error = q - jnp.expand_dims(target_q, -1)
            # Better bootstrapping for truncated episodes.
            truncation = transitions.extras['state_extras']['truncation']
            q_error *= jnp.expand_dims(1 - truncation, -1)
            critic_loss = 0.5 * jnp.mean(jnp.square(q_error))

            metric = {
                "q_error" : jnp.mean( q_error ),
                "buffer_reward" : jnp.mean( reward ),
                "target_q" : jnp.mean( target_q ),
                "target_q_min" : jnp.min( target_q ),
                "target_q_max" :  jnp.max( target_q ),
            }

            return critic_loss, metric
            
        (loss, metrics), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params,
            training_state.critic_state.target_params,
            training_state.actor_state.params,
            training_state.normalizer_params,
            transitions, 
            key
        )

        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        new_critic_state = new_critic_state.replace(
            target_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - args.tau) + y * args.tau,
                new_critic_state.target_params,
                new_critic_state.params,
            )
        )
        training_state = training_state.replace(critic_state = new_critic_state)

        return training_state, metrics
    
    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        return (training_state, key,), metrics

    @jax.jit
    def learn_step(training_state, buffer_state, key):
        experience_key, training_key = jax.random.split(key)

        # sample actor-step worth of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # process transitions for training
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key, len(transitions.action))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        # take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
        
        return training_state, buffer_state, metrics
    
    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    xt = time.time()
    metrics = None

    print('prefilling replay buffer....')
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    for ts in range(1, args.num_training_step + 1):
        
        key_sgd, key_generate_rollout, key = jax.random.split(key, 3)

        data_collect_start = time.time()
        training_state, env_state, buffer_state, data_metrics = data_collect_step(training_state, env_state, buffer_state, key_generate_rollout)
        data_collect_step_time += time.time() - data_collect_start
        
        learn_step_start = time.time()
        training_state, buffer_state, training_metrics = learn_step(training_state, buffer_state, key_sgd)
        learn_step_time += time.time() - learn_step_start

        if metrics is None:
            metrics = data_metrics | training_metrics
        else:
            metrics = jax.tree_util.tree_map(
                lambda x, y: x + y, metrics, (data_metrics | training_metrics)
            )

        if ts % args.num_training_steps_per_eval == 0:
            es = ts // args.num_training_steps_per_eval
            
            metrics = jax.tree_util.tree_map(
                lambda x: x / args.num_training_steps_per_eval, metrics
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            training_step_time = time.time() - xt            
            training_walltime += training_step_time

            sps = (
                args.num_training_steps_per_eval
                * args.num_envs * args.rollout_length
            ) / training_step_time

            metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                'training/data_collection_time_fraction' : data_collect_step_time / training_step_time,
                'training/learning_time_fraction' : learn_step_time / training_step_time,
                'training/env_steps': training_state.env_steps,
                'normalizer/count' : training_state.normalizer_params.count,
                'normalizer/mean' : jnp.mean( training_state.normalizer_params.mean ),
                'normalizer/summer_variance' : jnp.mean( training_state.normalizer_params.summed_variance ),
                'normalizer/std' : jnp.mean( training_state.normalizer_params.std ),
                **{f'training/{name}': value for name, value in metrics.items()},
                'buffer_current_size': replay_buffer.size(buffer_state),
            }

            rollout.shutdown_persistent_pool()
            metrics = evaluator.run_evaluation(
                policy_params={'policy':training_state.actor_state.params, 'normalizer':training_state.normalizer_params,},
                training_metrics=metrics,
            )
            rollout.shutdown_persistent_pool()

            pprint.pprint(metrics)
            if args.track:
                wandb.log(metrics, step=es)
                if args.wandb_mode == 'offline':
                    trigger_sync()
            metrics = None

            if args.save_checkpoint:
                save_params(
                    f"{save_path}/params_{es}.pkl", 
                    params = (
                        training_state.actor_state.params,
                        training_state.critic_state.params,
                        training_state.normalizer_params,
                    )
                )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        wandb.finish()
            
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)