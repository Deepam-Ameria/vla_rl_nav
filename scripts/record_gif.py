"""
Record an evaluation episode as a GIF.

Usage:
  python scripts/record_gif.py --model models/ppo_mujoco.zip --target red
  python scripts/record_gif.py --random-policy --target blue --out renders/random.gif
"""

import argparse
import os
import numpy as np
import imageio
import gymnasium as gym
import env.mujoco_nav_env


def record(model_path, target_name, out_path, random_policy=False):
    env = gym.make("MuJoCoNavEnv-v0", render_mode="rgb_array", target_name=target_name)
    obs, info = env.reset(seed=0)

    if not random_policy:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")

    frames = [env.render()]
    done = False

    while not done:
        if random_policy:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        done = terminated or truncated

    env.close()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=20)
    success = info.get("success", False)
    print(f"Saved {len(frames)} frames → {out_path}  |  success={success}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_mujoco.zip")
    parser.add_argument("--target", type=str, default=None,
                        help="red | blue | green | yellow | None (random)")
    parser.add_argument("--out", type=str, default="renders/eval.gif")
    parser.add_argument("--random-policy", action="store_true")
    args = parser.parse_args()
    record(args.model, args.target, args.out, args.random_policy)
