import argparse
import env.nav_env
import gymnasium as gym
import wandb

import env.nav_env
import env.mujoco_nav_env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import KVWriter, HumanOutputFormat, Logger


# ---------------------------------------------------------------------------
# WandB logger — plugs directly into SB3's internal logging system.
#
# SB3 uses a Logger that holds a list of "output formats" (KVWriter objects).
# Every time SB3 records a metric (entropy, KL, value_loss, ep_rew_mean, ...)
# it calls logger.dump() which calls write() on each KVWriter.
# By adding our WandbWriter to that list, ALL SB3 metrics flow to WandB
# automatically — no manual extraction needed.
# ---------------------------------------------------------------------------
class WandbWriter(KVWriter):
    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, str],
        step: int = 0,
    ) -> None:
        # Only log scalar numerics; skip strings and excluded keys
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

class WandbCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                wandb.log({
                    "episode_reward": info["episode"]["r"],
                    "episode_length": info["episode"]["l"],
                    "success": float(info.get("success", 0)),
                    "timestep": self.num_timesteps,
                })
        return True


def main(args):
    wandb.init(
        project="vla-rl-nav",
        name=args.run_name,
        config={
            "algo": "PPO",
            "total_timesteps": args.timesteps,
            "n_envs": args.n_envs,
            "device": args.device,
        }
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
        verbose=1,
        tensorboard_log="./logs/",
        device=args.device,
    )
    model.set_logger(make_logger())

    model.learn(
        total_timesteps=args.timesteps,
        callback=WandbCallback(),
    )

    model.save(args.save_path)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="ppo_baseline")
    parser.add_argument("--save-path", type=str, default="ppo_nav_model")
    args = parser.parse_args()
    main(args)
