"""
MuJoCo Navigation Environment

A differential-drive robot navigates to a target zone in a 10x10 arena.
Four colored targets (red/blue/green/yellow) are fixed in the scene.
Each episode picks one target; the agent must reach it.

MuJoCo concepts used here:
  - MjModel: static description loaded from XML (geometry, joints, actuators)
  - MjData:  dynamic state at runtime (qpos=positions, qvel=velocities, etc.)
  - freejoint: gives a body 6-DOF freedom; qpos stores [x, y, z, qw, qx, qy, qz]
  - mj_forward(): recompute all derived quantities (e.g. for rendering) without
                  stepping time — call this after manually setting qpos
  - Renderer:  mujoco.Renderer captures rgb frames for GIFs and evaluation plots
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

import mujoco

# Path to the XML file defining the scene
_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "nav_arena.xml")

TARGETS = {
    "red":    np.array([ 2.5,  2.5]),
    "blue":   np.array([-2.5,  2.5]),
    "green":  np.array([ 2.5, -2.5]),
    "yellow": np.array([-2.5, -2.5]),
}


class MuJoCoNavEnv(gym.Env):
    """
    Observation (5-dim):  [x, y, theta, dx_robot, dy_robot]
      - (x, y):              robot position in world frame
      - theta:               robot heading
      - (dx_robot, dy_robot): vector to active target, rotated into robot frame
                              (i.e. goal direction relative to where robot is pointing)

    Action (2-dim, continuous): [v, omega]
      - v:     forward speed  in [-1, 1]
      - omega: angular speed  in [-1, 1]

    Reward: -dist + 0.3 * alignment
      The alignment term gives early credit for pointing toward the goal,
      which fixes a credit-assignment problem: without it, the agent learns
      curved trajectories because rotating has no immediate reward.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, target_name=None):
        super().__init__()

        # Load model and allocate data buffers
        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.world_size = 5.0
        self.max_steps = 400
        self.current_step = 0
        self.render_mode = render_mode

        # Which target to navigate to.  None = random each episode.
        self.target_name = target_name
        self.active_target_name = None
        self.active_target_pos = None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # Renderer is created lazily (only if render() is called)
        self._renderer = None

        # Cache body ids for fast lookup — mujoco.mj_name2id is the API for this
        self._robot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robot"
        )
        self._target_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"target_{name}")
            for name in TARGETS
        }

        # Cache the freejoint qpos address once
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot_joint")
        self._qpos_addr = self.model.jnt_qposadr[joint_id]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick which target is active this episode
        if self.target_name is None:
            self.active_target_name = self.np_random.choice(list(TARGETS.keys()))
        else:
            self.active_target_name = self.target_name
        self.active_target_pos = TARGETS[self.active_target_name].copy()

        # Randomise robot start pose
        x = self.np_random.uniform(-4, 4)
        y = self.np_random.uniform(-4, 4)
        theta = self.np_random.uniform(-np.pi, np.pi)

        self._state = np.array([x, y, theta, 0.0, 0.0], dtype=np.float64)
        self.current_step = 0

        # Write state into MuJoCo data so the renderer sees the right pose
        self._sync_mujoco()

        return self._get_obs(), {"target": self.active_target_name}

    def step(self, action):
        x, y, theta, _, _ = self._state
        v_cmd = float(np.clip(action[0], -1.0, 1.0))
        omega_cmd = float(np.clip(action[1], -1.0, 1.0))

        dt = 0.05  # must match <option timestep> in the XML

        # Differential drive kinematics
        x += v_cmd * np.cos(theta) * dt
        y += v_cmd * np.sin(theta) * dt
        theta += omega_cmd * dt * 4.0
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self._state = np.array([x, y, theta, v_cmd, omega_cmd], dtype=np.float64)
        self.current_step += 1

        self._sync_mujoco()

        # --- Reward ---
        goal = self.active_target_pos
        dx = goal[0] - x
        dy = goal[1] - y
        dist = np.sqrt(dx * dx + dy * dy)

        dx_robot = np.cos(theta) * dx + np.sin(theta) * dy
        dy_robot = -np.sin(theta) * dx + np.cos(theta) * dy
        theta_error = np.arctan2(dy_robot, dx_robot)
        alignment = np.cos(theta_error)

        reward = -dist + 0.3 * alignment

        success = dist < 0.5
        terminated = success
        if success:
            reward += 10.0

        truncated = self.current_step >= self.max_steps

        if np.abs(x) > self.world_size or np.abs(y) > self.world_size:
            reward -= 10.0
            terminated = True

        info = {}
        if terminated or truncated:
            info["success"] = success
            info["target"] = self.active_target_name

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=480)
        self._renderer.update_scene(self.data, camera="top_down")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        x, y, theta, _, _ = self._state
        goal = self.active_target_pos
        dx = goal[0] - x
        dy = goal[1] - y
        dx_robot = np.cos(theta) * dx + np.sin(theta) * dy
        dy_robot = -np.sin(theta) * dx + np.cos(theta) * dy
        return np.array([x, y, theta, dx_robot, dy_robot], dtype=np.float32)

    def _sync_mujoco(self):
        """Write robot pose from _state into MuJoCo's qpos, then call mj_forward.

        The robot has a freejoint, so its qpos slice is:
          [x, y, z,  qw, qx, qy, qz]   (positions then quaternion)
        We keep z fixed at 0.12 (robot height) and rotate only around Z-axis.
        """
        x, y, theta = self._state[0], self._state[1], self._state[2]

        self.data.qpos[self._qpos_addr:self._qpos_addr + 3] = [x, y, 0.12]
        # Rotation around Z by theta: quaternion = [cos(t/2), 0, 0, sin(t/2)]
        self.data.qpos[self._qpos_addr + 3] = np.cos(theta / 2)  # qw
        self.data.qpos[self._qpos_addr + 4] = 0.0                 # qx
        self.data.qpos[self._qpos_addr + 5] = 0.0                 # qy
        self.data.qpos[self._qpos_addr + 6] = np.sin(theta / 2)  # qz

        # mj_forward recomputes body positions, velocities, etc. from qpos
        # without advancing simulation time — essential for correct rendering
        mujoco.mj_forward(self.model, self.data)


register(
    id="MuJoCoNavEnv-v0",
    entry_point="env.mujoco_nav_env:MuJoCoNavEnv",
)
