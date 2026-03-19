import argparse
import env.nav_env
import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class WandbCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
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
            env = gym.make("NavEnv-v0")
            return Monitor(env)
        return _init

    env = DummyVecEnv([make_env() for _ in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        device=args.device,
    )

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
