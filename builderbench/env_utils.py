import re

import jax
from flax import struct

from etils import epath
from typing import Any, Dict

from builderbench.constants import _TIME_STEPS

@struct.dataclass
class State:
    """Environment state for training and inference."""
    physics_state: jax.Array                         # minimal bits of information that are necessary to run the simulation
    sensordata: jax.Array                            # data returned by various sensors
    ctrl: jax.Array                                  # target values for the actuators 
    obs: jax.Array                                   # observation for the RL agent
    reward: jax.Array                                # reward for the RL agent
    done: jax.Array                                  # terminal signal for the RL agent
    metrics: Dict[str, jax.Array]                    # metrics to store during training 
    info: Dict[str, Any]                             # misc information

def get_assets(xml_dir_path) -> Dict[str, bytes]:
    assets = {}

    path = epath.Path(xml_dir_path)
    
	# add all xmls
    glob = '*.xml'
    for f in epath.Path(path).glob(glob):
        if f.is_file():
            assets[f.name] = f.read_bytes()

    # add all files (obj & stl) inside assets/
    path = path / 'assets'
    for f in epath.Path(path).glob('*'):
        if f.is_file():
            assets[f.name] = f.read_bytes()

    return assets

def make_env(args):
    # Initialize environment
    if re.fullmatch(r"cube-\d+-task\d+", args.env_id):
        num_cubes = int( re.search(r"cube-(\d+)", args.env_id).group(1))
        task_id = int( re.search(r"task(\d+)", args.env_id).group(1)) - 1

        from builderbench.build_block import CreativeCube, default_config
        env_class = CreativeCube
        default_config = default_config()
        default_config.num_cubes = num_cubes
        default_config.task_id = task_id
        episode_length = 100 + num_cubes * 50
    
    elif re.fullmatch(r"cube-\d+-play", args.env_id):
        num_cubes = int( re.search(r"cube-(\d+)", args.env_id).group(1))

        from builderbench.build_block_play import CreativeCubePlay, default_config
        env_class = CreativeCubePlay
        default_config = default_config()
        default_config.num_cubes = num_cubes
        episode_length = num_cubes * 500

    else:
        raise ValueError(f"Environment {args.env_id} not supported")

    default_config.episode_length = episode_length
    default_config.sim_dt, default_config.ctrl_dt = _TIME_STEPS[args.env_id]
    default_config.env_early_termination = args.env_early_termination
    default_config.permutation_invariant_reward = args.permutation_invariant_reward
    
    return env_class, default_config