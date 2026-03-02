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
                    "success": float(info.get("success", 0))
                })

        return True


def main():
    wandb.init(
        project="vla-rl-nav",
        name="ppo_phase1_baseline",
        config={
            "algo": "PPO",
            "total_timesteps": 300_000
        }
    )

    # env = gym.make("NavEnv-v0")
    # env = Monitor(env)


    def make_env():
        def _init():
            env = gym.make("NavEnv-v0")
            return Monitor(env)
        return _init

    env = DummyVecEnv([make_env() for _ in range(8)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        device = "cpu"
    )

    model.learn(
        total_timesteps=100_000,
        callback=WandbCallback()
    )

    model.save("ppo_nav_model")
    wandb.finish()


if __name__ == "__main__":
    main()