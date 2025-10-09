import os
import jax
import tyro
import mediapy
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from utils.networks import load_params

AGENT = "ppo"

if AGENT == "ppo":
    from ppo import Args as PPOArgs
    @dataclass
    class Args(PPOArgs):
        folder_path: str = "checkpoints/"
        fps: int = 10

else:
    raise NotImplementedError

def main(args: Args):

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    if args.agent in ["ppo"]:
        
        from builderbench.env_utils import make_env
        from utils.wrapper import wrap_env
        env_class, default_config = make_env(args)
        env = wrap_env( env_class(num_envs=1, num_threads=1, config=default_config), default_config.episode_length )   
        action_size = env.action_size

        from utils.evaluation import get_video

        from ppo import PPONetworks, Actor, Value
        ppo_network = PPONetworks( 
            policy_network = Actor(layer_sizes=args.policy_hidden_sizes + [action_size * 2]),
        value_network = Value(layer_sizes=args.value_hidden_sizes  + [1]),
        )

        from ppo import make_inference_fn
        make_policy = make_inference_fn(ppo_network)

    else:
        raise NotImplementedError

    folder_path = Path(args.folder_path)
    
    for subfolder in Path(folder_path).iterdir():
        if subfolder.is_dir() and args.env_id in subfolder.name:
            print(f"\nSubfolder: {subfolder}")

            video_path = f"{folder_path.parent}/videos/{subfolder.name}/"
            os.makedirs(video_path, exist_ok=True)

            for param_file in subfolder.iterdir():  

                if not Path( f"{video_path}/{param_file.stem}.mp4" ).exists(): 

                    params = load_params(f"{param_file}")
                    actor_params, _, normalize_params = params

                    jit_inference_fn = jax.jit(
                                        make_policy(
                                            {
                                                'policy': actor_params, 
                                                'normalizer': normalize_params,
                                            },
                                            deterministic=True,
                                        )
                                    )

                    key, video_key = jax.random.split(key)
                    video_images = get_video(args.env_id, jit_inference_fn, env, video_key, default_config.episode_length)
                    mediapy.write_video(f"{video_path}/{param_file.stem}.mp4", video_images, fps=args.fps)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
