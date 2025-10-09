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
import mediapy
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
from utils.wrapper import wrap_env
from utils.evaluation import Evaluator
from utils.networks import MLP, save_params
from builderbench.env_utils import make_env


@dataclass
class Args:
    # experiment
    agent: str = "ppo-rnd"
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
    rollout_length: int = 160
    num_minibatches_per_rollout: int = 32
    num_epochs_per_rollout: int = 8    
    learning_rate: float = 1e-4
    discount: float = 0.99
    int_discount: float = 0.99
    int_loss_cost: float = 1.0
    ext_loss_cost: float = 2.0
    entropy_cost: float = 2e-2
    reward_scaling: float = 1.0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    normalize_advantage: bool = True

@flax.struct.dataclass
class PPOTrainingState(TrainState):
  """Contains training state for the learner."""
  int_reward_normalizer_params: Any
  normalizer_params: Any
  env_steps: float

class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()

@flax.struct.dataclass
class PPONetworks:
    policy_network: Any
    value_network: Any
    int_value_network: Any
    rnd_network: Any

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

class Value(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = False
    
    def setup(self):
        self.value_net = MLP(self.layer_sizes, activation=self.activation, layer_norm=self.layer_norm)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
        value = self.value_net(x)
        return jnp.squeeze(value, axis=-1)

class RND(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = False

    def setup(self):
        self.prediction_net = MLP(self.layer_sizes, activation=self.activation, layer_norm=self.layer_norm)
        self.target_net = MLP(self.layer_sizes, activation=self.activation, layer_norm=self.layer_norm)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
            x = jnp.clip(x, a_max=5, a_min=-5)
        prediction = self.prediction_net(x)
        target = jax.lax.stop_gradient( (self.target_net(x)) )
        return prediction, target

def make_inference_fn(ppo_networks):
    """Creates params and inference function for the PPO agent."""
    def make_policy(params, deterministic: bool = False):
        policy_network = ppo_networks.policy_network
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
    args.minibatch_size = args.num_envs * args.rollout_length // ( args.num_minibatches_per_rollout )
        
    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = {args.num_minibatches_per_rollout * args.num_epochs_per_rollout}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")
    print(f"Data to update ratio = {  ( args.num_envs * args.rollout_length ) / (args.num_minibatches_per_rollout * args.num_epochs_per_rollout)}")    

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
        
    key = jax.random.PRNGKey(args.seed)
    local_key, key_env, key_eval, key_policy, key_value, key_rnd = jax.random.split(key, 6)

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
    key_envs = jax.random.split(key_env, args.num_envs)
    env_state = reset_fn(key_envs)
    obs_size = env.observation_size
    action_size = env.action_size
    goal_size = env.goal_size

    log_data_metric_keys = []
    for k in ("obj_reached_once", "obj_lifted", "obj_moved"):
        if k in env_state.metrics.keys():
            log_data_metric_keys.append(k)
    log_data_metric_keys = tuple(log_data_metric_keys)

    # Initialize PPO networks
    ppo_network = PPONetworks( 
        policy_network = Actor(layer_sizes=[256,] * 4 + [action_size * 2]),
        value_network = Value(layer_sizes=[256,] * 4  + [1]),
        int_value_network = Value(layer_sizes=[256,] * 4  + [1]),
        rnd_network = RND(layer_sizes=[256,] * 4),
    )
    training_state = PPOTrainingState.create(
        apply_fn=None,
        params={
            'policy': ppo_network.policy_network.init( key_policy, x=jnp.zeros((1, obs_size+goal_size)) ),
            'value': ppo_network.value_network.init( key_value, x=jnp.zeros((1, obs_size+goal_size)) ),
            'int_value': ppo_network.int_value_network.init( key_value, x=jnp.zeros((1, obs_size+goal_size)) ),
            'rnd': ppo_network.rnd_network.init( key_value, x=jnp.zeros((1, obs_size+goal_size)) ),
        },
        tx=optax.adam(learning_rate=args.learning_rate),  
        normalizer_params=running_statistics.init_state((obs_size+goal_size,) ),
        int_reward_normalizer_params=running_statistics.init_state(()),
        env_steps=np.zeros((), dtype=np.float64),
    )
    make_policy = make_inference_fn(ppo_network)

    # Initialize evaluators
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=key_eval,
    )

    def generate_unroll(
        env,
        env_state,
        policy,
        key,
        unroll_length,
        extra_fields,
    ):
        """Collect trajectories of given unroll_length."""        
        @jax.jit
        def f(carry, unused_t):
            env_state, key = carry
            key, next_key = jax.random.split(key)
            actions, policy_extras = policy(env_state.obs, env_state.info['target_goal'], key)  
            
            next_env_state = env.pre_step(env_state, actions)  
            physics_state, sensor_data = env.step(next_env_state, actions)
            next_env_state = env.post_step(next_env_state, physics_state, sensor_data)

            state_extras = {x: next_env_state.info[x] for x in extra_fields}

            metrics = {x: next_env_state.metrics[x] for x in log_data_metric_keys}

            transition = Transition(
                observation=jnp.concatenate( [env_state.obs, env_state.info['target_goal']], axis=-1),
                action=actions,
                reward=next_env_state.reward,
                discount=1 - next_env_state.done,
                next_observation=jnp.concatenate( [next_env_state.obs, next_env_state.info['target_goal']], axis=-1),
                extras={'policy_extras': policy_extras, 'state_extras': state_extras},
            )
            
            return (next_env_state, next_key), (transition, metrics)

        (final_env_state, _), (data, data_metrics) = jax.lax.scan(
            f, (env_state, key), (), length=unroll_length
        )
        return final_env_state, data, data_metrics

    @jax.jit
    def data_collect_step(training_state, env_state, key_generate_rollout):
        policy = make_policy({
            'policy': training_state.params['policy'], 
            'normalizer': training_state.normalizer_params,
            })
        
        env_state, data, data_metrics = generate_unroll(
                env,
                env_state,
                policy,
                key_generate_rollout,
                args.rollout_length,
                extra_fields=('truncation',),
            )
        
        # Updating collected data with intrinsic rewards
        int_prediction, int_target = ppo_network.rnd_network.apply(training_state.params['rnd'], data.next_observation, training_state.normalizer_params)
        int_rewards = jnp.sum( (int_prediction - int_target) ** 2, axis=-1) / 2
        data.extras['policy_extras']['int_reward'] = int_rewards
        
        # Update normalization params.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
        )
        int_reward_normalizer_params = running_statistics.update(
            training_state.int_reward_normalizer_params,
            jnp.sum( int_rewards * ( args.int_discount ** jnp.arange(args.rollout_length)[:, None] ), axis=-1),
        )

        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            int_reward_normalizer_params=int_reward_normalizer_params,
            env_steps=training_state.env_steps + args.rollout_length * args.num_envs,
        )

        return training_state, env_state, data, data_metrics
    
    def compute_gae(
        truncation: jnp.ndarray,
        termination: jnp.ndarray,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        bootstrap_value: jnp.ndarray,
        lambda_: float = 1.0,
        discount: float = 0.99,
    ):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = jnp.concatenate(
            [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = jnp.zeros_like(bootstrap_value)
        vs_minus_v_xs = []

        def compute_vs_minus_v_xs(carry, target_t):
            lambda_, acc = carry
            truncation_mask, delta, termination = target_t
            acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
            return (lambda_, acc), (acc)

        (_, _), (vs_minus_v_xs) = jax.lax.scan(
            compute_vs_minus_v_xs,
            (lambda_, acc),
            (truncation_mask, deltas, termination),
            length=int(truncation_mask.shape[0]),
            reverse=True,
        )
        # Add V(x_s) to get v_s.
        vs = jnp.add(vs_minus_v_xs, values)

        vs_t_plus_1 = jnp.concatenate(
            [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        advantages = (
            rewards + discount * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


    def compute_ppo_loss(
        params,
        normalizer_params,
        init_normalizer_params,
        data,
        rng,
    ):
        bijector = distrax.Tanh()  
        policy_apply = ppo_network.policy_network.apply
        value_apply = ppo_network.value_network.apply
        int_value_apply = ppo_network.int_value_network.apply
        rnd_apply = ppo_network.rnd_network.apply

        data, value_targets, int_value_targets, advantages = data

        # Policy function loss
        policy_dist = policy_apply(params['policy'], data.observation, normalizer_params)
        target_action_log_probs = policy_dist.log_prob( data.extras['policy_extras']['raw_action'] ) - bijector.forward_log_det_jacobian( data.extras['policy_extras']['raw_action'] )
        target_action_log_probs = jnp.sum(target_action_log_probs, axis=-1)  
        behaviour_action_log_probs = data.extras['policy_extras']['log_prob']
        rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = (jnp.clip(rho_s, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon) * advantages)
        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Forwad loss
        predict_next_state_feature, target_next_state_feature = rnd_apply(params['rnd'], data.next_observation)
        forward_loss = jnp.mean( (predict_next_state_feature-target_next_state_feature)**2 )

        # Value function loss
        baseline = value_apply(params['value'], data.observation, normalizer_params)
        v_error = value_targets - baseline
        v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

        # Intrinsic Value function loss
        int_baseline = int_value_apply(params['int_value'], data.observation, normalizer_params)
        int_v_error = int_value_targets - int_baseline
        int_v_loss = jnp.mean(int_v_error * int_v_error) * 0.5 * 0.5

        # Entropy loss
        entropy = policy_dist.entropy() + bijector.forward_log_det_jacobian( policy_dist.sample(seed=rng) )
        entropy = jnp.mean( jnp.sum(entropy, axis=-1) )
        entropy_loss = args.entropy_cost * -entropy

        total_loss = policy_loss + v_loss + int_v_loss + entropy_loss + forward_loss
        return total_loss, {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'v_loss': v_loss,
            'int_v_loss': int_v_loss,
            'entropy_loss': entropy_loss,
            'forward_loss': forward_loss,
        }
    
    @jax.jit
    def learn_step(training_state, data, key_sgd):

        def _learn_step(carry, unused_t):
            
            def _train_minibatch_step(carry, data):
                training_state, key = carry
                key, key_loss = jax.random.split(key)
                
                (_, metrics), grads = jax.value_and_grad(compute_ppo_loss, has_aux=True)(training_state.params, training_state.normalizer_params, training_state.int_reward_normalizer_params, data, key_loss)
                training_state = training_state.apply_gradients(grads=grads)
                
                return (training_state, key), metrics
            
            training_state, data, value_targets, int_value_targets, advantages, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)
        
            def shuffle_and_reshape(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (args.num_minibatches_per_rollout, -1) + x.shape[2:])
                return x

            batch_data = ( data, value_targets, int_value_targets, advantages )
            shuffled_batch_data = jax.tree_util.tree_map(shuffle_and_reshape, batch_data)

            (training_state, _), metrics = jax.lax.scan(
                _train_minibatch_step,
                (training_state, key_grad),
                shuffled_batch_data,
                length=args.num_minibatches_per_rollout,
            )
            return (training_state, data, value_targets, int_value_targets, advantages, key), metrics

        # calculate gae
        baseline = ppo_network.value_network.apply(training_state.params['value'], data.observation, training_state.normalizer_params)
        int_baseline = ppo_network.int_value_network.apply(training_state.params['int_value'], data.observation, training_state.normalizer_params)

        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        bootstrap_value = ppo_network.value_network.apply(training_state.params['value'], terminal_obs, training_state.normalizer_params)
        int_bootstrap_value = ppo_network.int_value_network.apply(training_state.params['int_value'], terminal_obs, training_state.normalizer_params)

        rewards = data.reward * args.reward_scaling
        int_rewards = data.extras['policy_extras']['int_reward'] / training_state.int_reward_normalizer_params.std
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)

        value_targets, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=args.gae_lambda,
            discount=args.discount,
        )
        int_value_targets, int_advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=int_rewards,
            values=int_baseline,
            bootstrap_value=int_bootstrap_value,
            lambda_=args.gae_lambda,
            discount=args.int_discount,
        )

        advantages = int_advantages * args.int_loss_cost + advantages * args.ext_loss_cost
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        (training_state, _, _, _, _, _), metrics = jax.lax.scan(
            _learn_step,
            (training_state, data, value_targets, int_value_targets, advantages, key_sgd),
            (),
            length=args.num_epochs_per_rollout,
        )

        return training_state, metrics

    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    xt = time.time()
    metrics = None
    for ts in range(1, args.num_training_step + 1):
        
        key_sgd, key_generate_unroll, key = jax.random.split(key, 3)

        data_collect_start = time.time()
        training_state, env_state, training_data, data_metrics = data_collect_step(training_state, env_state, key_generate_unroll)
        data_collect_step_time += time.time() - data_collect_start
        
        learn_step_start = time.time()
        training_state, training_metrics = learn_step(training_state, training_data, key_sgd)
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
                'int_reward_normalizer/std' : jnp.mean( training_state.normalizer_params.std ),
                'int_reward_normalizer/count' : training_state.int_reward_normalizer_params.count,
                'int_reward_normalizer/mean' : jnp.mean( training_state.int_reward_normalizer_params.mean ),
                'int_reward_normalizer/summer_variance' : jnp.mean( training_state.int_reward_normalizer_params.summed_variance ),
                'int_reward_normalizer/std' : jnp.mean( training_state.int_reward_normalizer_params.std ),
                **{f'training/{name}': value for name, value in metrics.items()},
            }

            rollout.shutdown_persistent_pool()
            metrics = evaluator.run_evaluation(
                policy_params={'policy':training_state.params['policy'], 'normalizer':training_state.normalizer_params},
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
                        training_state.params,
                        training_state.normalizer_params,
                        training_state.int_reward_normalizer_params,
                    )
                )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        wandb.finish()
            
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)