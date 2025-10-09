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
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
from dataclasses import dataclass
from typing import NamedTuple
from wandb_osh.hooks import TriggerWandbSyncHook

from utils.buffer import TrajectoryUniformSamplingQueue
from utils.wrapper import wrap_env
from utils.evaluation import Evaluator
from utils.networks import MLP, save_params
from buildstuff.env_utils import make_env

@dataclass
class Args:
    # experiment
    agent: str = "crl"
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    
    # logging and checkpointing
    track: bool = False
    wandb_project_name: str = "buildstuff"
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
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    discount: float = 0.99
    entropy_cost: float = 0.1
    logsumexp_cost: float = 0.1
    rep_size: int = 64
    max_replay_size: int = 10000
    min_replay_size: int = 1000

@flax.struct.dataclass
class CRLTrainingState:
    """Contains training state for the learner"""
    env_steps: np.ndarray
    gradient_steps: np.ndarray
    actor_state: TrainState
    critic_state: TrainState
    
class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    achieved_goal: jnp.ndarray
    action: jnp.ndarray
    extras: jnp.ndarray = ()

class SA_encoder(nn.Module):
    rep_size: int
    norm_type = "layer_norm"
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.rep_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x
    
class G_encoder(nn.Module):
    rep_size: int
    norm_type = "layer_norm"
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(g)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.rep_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x
    
class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, s, g_repr):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = jnp.concatenate([s, g_repr], axis=-1)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std
    
def make_inference_fn(policy_network, g_encoder_network):
    """Creates params and inference function for the CRL agent."""
    def make_policy(params, deterministic: bool = False):

        def policy(observations, goals, key_sample):
            
            goals = g_encoder_network.apply(params['g_encoder'], goals)
            means, log_stds = policy_network.apply(params['actor'], observations, goals)
                
            if deterministic:
                return nn.tanh( means ), {}
            
            stds = jnp.exp(log_stds)
            raw_actions = means + stds * jax.random.normal(key_sample, shape=means.shape, dtype=means.dtype)
            postprocessed_actions = nn.tanh(raw_actions)
                
            log_prob = jax.scipy.stats.norm.logpdf(raw_actions, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(postprocessed_actions)) + 1e-6)
            log_prob = log_prob.sum(-1)

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
    key, buffer_key, env_key, eval_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

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
    actor = Actor(action_size=action_size)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size]), np.ones([1, args.rep_size])),
        tx=optax.adam(learning_rate=args.actor_learning_rate)
    )

    # Critic
    sa_encoder = SA_encoder(rep_size=args.rep_size)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, obs_size]), np.ones([1, action_size]))
    g_encoder = G_encoder(rep_size=args.rep_size)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_size]))
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=optax.adam(learning_rate=args.critic_learning_rate),
    )

    actor.apply = jax.jit(actor.apply)
    sa_encoder.apply = jax.jit(sa_encoder.apply)
    g_encoder.apply = jax.jit(g_encoder.apply)

    # Trainstate
    training_state = CRLTrainingState(
        env_steps=np.zeros((), dtype=np.float64),
        gradient_steps=np.zeros((), dtype=np.float64),
        actor_state=actor_state,
        critic_state=critic_state,
    )

    #Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_goal = jnp.zeros((goal_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        achieved_goal=dummy_goal,
        action=dummy_action,
        extras={
            "state_extras": {
                "traj_id": 0.0,
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
                sequence_length=args.sequence_length+1,
            )
        )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    make_policy = make_inference_fn(actor, g_encoder)
    # Initialize evaluators
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=eval_key,
    )

    def actor_step(training_state, env, env_state, key, extra_fields, metrics_fields):
        g_encoder_params = training_state.critic_state.params["g_encoder"]

        g_repr = g_encoder.apply(g_encoder_params, env_state.info['target_goal'])

        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs, g_repr)
        stds = jnp.exp(log_stds)
        actions = nn.tanh( means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype) )
        
        nstate = env.pre_step(env_state, actions)  
        physics_state, sensor_data = env.step(nstate, actions)
        nstate = env.post_step(nstate, physics_state, sensor_data)
        
        state_extras = {x: nstate.info[x] for x in extra_fields}
        metrics = {x: nstate.metrics[x] for x in metrics_fields}
        
        return training_state, nstate, Transition(
                                            observation=env_state.obs,
                                            achieved_goal=env_state.info['achieved_goal'],
                                            action=actions,
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
                extra_fields=("traj_id",),
                metrics_fields=log_data_metric_keys,
            )
            return (training_state, env_state, next_key), (transition, metrics)

        (training_state, env_state, _), (data, metrics) = jax.lax.scan(f, (training_state, env_state, key), (), length=args.rollout_length)

        training_state = training_state.replace(
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
            state = transitions.observation
            goal = transitions.extras['future_goal']
            sa_encoder_params, g_encoder_params = jax.lax.stop_gradient(critic_params["sa_encoder"]), jax.lax.stop_gradient(critic_params["g_encoder"])
            
            g_repr = g_encoder.apply(g_encoder_params, goal)

            means, log_stds = actor.apply(actor_params, state, g_repr)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)           # dimension = B

            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            actor_loss = jnp.mean( args.entropy_cost * log_prob - (qf_pi) )

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
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            
            state = transitions.observation
            action = transitions.action
            goal = transitions.extras['future_goal']
            
            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            g_repr = g_encoder.apply(g_encoder_params, goal)
            
            # InfoNCE
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)) #shape = BxB

            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1)) 

            # logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss += args.logsumexp_cost * ( jnp.mean(logsumexp**2) )

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

            return critic_loss, (logsumexp, correct, logits_pos, logits_neg)
            
        (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state = new_critic_state)

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
        }

        return training_state, metrics
    
    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        training_state = training_state.replace(gradient_steps = training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        return (training_state, key,), metrics

    @jax.jit
    def learn_step(training_state, buffer_state, key):
        experience_key, sampling_key, training_key = jax.random.split(key, 3)

        # sample actor-step worth of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.discount,), transitions, batch_keys
        )
        
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
                **{f'training/{name}': value for name, value in metrics.items()},
                'buffer_current_size': replay_buffer.size(buffer_state),
            }

            rollout.shutdown_persistent_pool()
            metrics = evaluator.run_evaluation(
                policy_params={"actor": training_state.actor_state.params, "g_encoder": training_state.critic_state.params["g_encoder"]},
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
                    )
                )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        wandb.finish()
            
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)