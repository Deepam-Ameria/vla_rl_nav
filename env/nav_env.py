import gymnasium as gym
from gymnasium import spaces
import numpy as np


class NavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.world_size = 5.0
        self.max_steps = 400
        self.current_step = 0

        # Observation: [x, y, theta, v, omega, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        # Action: [v, omega]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.state = None
        self.goal = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        theta = np.random.uniform(-np.pi, np.pi)

        self.state = np.array([x, y, theta, 0.0, 0.0], dtype=np.float32)
        self.goal = np.random.uniform(-4, 4, size=(2,))
        self.current_step = 0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.state, self.goal]).astype(np.float32)

    def step(self, action):
        x, y, theta, v, omega = self.state
        v_cmd, omega_cmd = action

        dt = 0.05

        # Differential drive kinematics
        x += v_cmd * np.cos(theta) * dt
        y += v_cmd * np.sin(theta) * dt
        theta += omega_cmd * dt

        self.state = np.array([x, y, theta, v_cmd, omega_cmd], dtype=np.float32)
        self.current_step += 1

        # Distance reward
        dist = np.linalg.norm(self.goal - self.state[:2])
        reward = -dist

        terminated = dist < 0.2
        truncated = self.current_step >= self.max_steps

        # Out of bounds penalty
        if np.abs(x) > self.world_size or np.abs(y) > self.world_size:
            reward -= 10.0
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}