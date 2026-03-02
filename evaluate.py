import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import env.nav_env


def run_episode(model, render_plot=True):
    env = gym.make("NavEnv-v0")
    obs, _ = env.reset()

    trajectory = []
    goal = env.unwrapped.goal.copy()
    done = False

    while not done:
        # Deterministic action (no exploration noise)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        x, y = env.unwrapped.state[0], env.unwrapped.state[1]
        trajectory.append([x, y])

        done = terminated or truncated

    trajectory = np.array(trajectory)

    if render_plot:
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="trajectory")
        plt.scatter(goal[0], goal[1], marker="*", s=200, label="goal")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.legend()
        plt.title("Evaluation Trajectory")
        plt.show()

    print("Success:", info.get("success", False))


if __name__ == "__main__":
    model = PPO.load("ppo_nav_model", device="cpu")
    run_episode(model)