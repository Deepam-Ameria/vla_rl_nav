import argparse
import sys
from typing import Any, Dict

import numpy as np
import gymnasium as gym
import wandb

import env.nav_env
import env.mujoco_nav_env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import KVWriter, HumanOutputFormat, Logger


class WandbWriter(KVWriter):
    """Plugs into SB3's logger so all training metrics stream to WandB."""
    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, str], step: int = 0) -> None:
        loggable = {
            k: float(v)
            for k, v in key_values.items()
            if isinstance(v, (int, float, np.floating, np.integer))
            and k not in key_excluded
        }
        if loggable:
            wandb.log(loggable, step=step)

    def close(self) -> None:
        pass


class EpisodeCallback(BaseCallback):
    """Logs per-episode metrics: reward, length, success, target."""
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                wandb.log({
                    "episode/reward":  info["episode"]["r"],
                    "episode/length":  info["episode"]["l"],
                    "episode/success": float(info.get("success", 0)),
                    "timestep":        self.num_timesteps,
                    **( {"episode/target": info["target"]} if "target" in info else {} ),
                })
        return True


def make_logger() -> Logger:
    return Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), WandbWriter()],
    )


def main(args):
    wandb.init(
        project="vla-rl-nav",
        name=args.run_name,
        config={
            "algo":            "PPO",
            "env":             args.env,
            "total_timesteps": args.timesteps,
            "n_envs":          args.n_envs,
            "device":          args.device,
        },
    )

    def make_env():
        def _init():
            e = gym.make(args.env)
            return Monitor(e)
        return _init

    env = DummyVecEnv([make_env() for _ in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=None,
        device=args.device,
    )
    model.set_logger(make_logger())

    model.learn(total_timesteps=args.timesteps, callback=EpisodeCallback())

    model.save(args.save_path)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",       type=str, default="MuJoCoNavEnv-v0",
                        choices=["NavEnv-v0", "MuJoCoNavEnv-v0"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs",    type=int, default=8)
    parser.add_argument("--device",    type=str, default="cpu")
    parser.add_argument("--run-name",  type=str, default="ppo_baseline")
    parser.add_argument("--save-path", type=str, default="models/ppo_mujoco")
    args = parser.parse_args()
    main(args)
