"""
Multi-rollout trajectory visualization — like the figures in research papers.

Runs N evaluation episodes and plots all trajectories overlaid on a top-down
view of the room, color-coded by success/failure.

Usage:
    python scripts/visualize_rollouts.py --model models/ppo_mujoco.zip
    python scripts/visualize_rollouts.py --model models/ppo_mujoco.zip --target red --n-rollouts 20
    python scripts/visualize_rollouts.py --model models/ppo_mujoco.zip --all-targets --n-rollouts 10
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import gymnasium as gym

import env.mujoco_nav_env
from env.mujoco_nav_env import TARGETS


# Room layout: where each furniture piece sits and its label
FURNITURE = {
    "red":    {"label": "Table",  "pos": ( 2.5,  2.5), "color": "#e02020"},
    "blue":   {"label": "Chair",  "pos": (-2.5,  2.5), "color": "#2060e0"},
    "green":  {"label": "Vase",   "pos": ( 2.5, -2.5), "color": "#20c040"},
    "yellow": {"label": "TV",     "pos": (-2.5, -2.5), "color": "#e0c010"},
}


def run_episode(model, env_instance, deterministic=True):
    """Run one episode, return trajectory and outcome."""
    obs, info = env_instance.reset()
    trajectory = [env_instance.unwrapped._state[:2].copy()]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env_instance.step(action)
        trajectory.append(env_instance.unwrapped._state[:2].copy())
        done = terminated or truncated

    return np.array(trajectory), info.get("success", False), info.get("target", None)


def plot_room_background(ax):
    """Draw the room footprint: walls, target zones, labels."""
    # Room boundary
    room = plt.Rectangle((-5, -5), 10, 10, linewidth=2,
                          edgecolor="#555", facecolor="#f0ece4")
    ax.add_patch(room)

    # Target zones
    for name, f in FURNITURE.items():
        x, y = f["pos"]
        circle = plt.Circle((x, y), 0.55, color=f["color"], alpha=0.25, zorder=2)
        ax.add_patch(circle)
        # Icon marker
        ax.plot(x, y, marker="*", markersize=14, color=f["color"],
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        # Label
        ax.text(x, y - 0.85, f["label"], ha="center", va="top",
                fontsize=8, color=f["color"], fontweight="bold")

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f0ece4")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def visualize_single_target(model, target_name, n_rollouts, out_path, deterministic):
    """One panel showing all rollouts for a single target."""
    env = gym.make("MuJoCoNavEnv-v0", target_name=target_name)
    f = FURNITURE[target_name]

    trajectories = []
    successes = []
    for _ in range(n_rollouts):
        traj, success, _ = run_episode(model, env, deterministic)
        trajectories.append(traj)
        successes.append(success)

    env.close()

    success_rate = sum(successes) / len(successes)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_room_background(ax)

    for traj, success in zip(trajectories, successes):
        color = f["color"] if success else "#888888"
        alpha = 0.7 if success else 0.35
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha,
                linewidth=1.2, zorder=4)
        # Start dot
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=4,
                alpha=alpha, zorder=5)
        # End marker
        marker = "^" if success else "x"
        ax.plot(traj[-1, 0], traj[-1, 1], marker, color=color, markersize=6,
                alpha=alpha, zorder=5)

    ax.set_title(
        f"Target: {f['label']}  |  {n_rollouts} rollouts  |  "
        f"Success: {success_rate:.0%}",
        fontsize=11, pad=10,
    )

    success_patch = mpatches.Patch(color=f["color"], label="Success")
    fail_patch    = mpatches.Patch(color="#888888",   label="Failure")
    ax.legend(handles=[success_patch, fail_patch], loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}  (success={success_rate:.0%})")


def visualize_all_targets(model, n_rollouts, out_path, deterministic):
    """2x2 grid, one panel per target — the research-paper style figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    fig.suptitle(f"PPO Navigation Policy — {n_rollouts} rollouts per target",
                 fontsize=14, fontweight="bold", y=0.98)

    target_order = [("red", axes[0, 0]),
                    ("blue", axes[0, 1]),
                    ("green", axes[1, 0]),
                    ("yellow", axes[1, 1])]

    for target_name, ax in target_order:
        f = FURNITURE[target_name]
        env = gym.make("MuJoCoNavEnv-v0", target_name=target_name)

        successes = []
        for _ in range(n_rollouts):
            traj, success, _ = run_episode(model, env, deterministic)
            successes.append((traj, success))

        env.close()
        success_rate = sum(s for _, s in successes) / n_rollouts

        plot_room_background(ax)

        for traj, success in successes:
            color = f["color"] if success else "#aaaaaa"
            alpha = 0.65 if success else 0.3
            ax.plot(traj[:, 0], traj[:, 1], color=color,
                    alpha=alpha, linewidth=1.0, zorder=4)
            ax.plot(traj[0, 0], traj[0, 1], "o", color=color,
                    markersize=3, alpha=alpha, zorder=5)
            marker = "^" if success else "x"
            ax.plot(traj[-1, 0], traj[-1, 1], marker, color=color,
                    markersize=5, alpha=alpha, zorder=5)

        ax.set_title(f"{f['label']}  —  {success_rate:.0%} success",
                     fontsize=10, color=f["color"], fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="models/ppo_mujoco.zip")
    parser.add_argument("--target",      type=str, default=None,
                        choices=list(FURNITURE.keys()))
    parser.add_argument("--all-targets", action="store_true",
                        help="2x2 grid with all 4 targets")
    parser.add_argument("--n-rollouts",  type=int, default=15)
    parser.add_argument("--out",         type=str, default=None)
    parser.add_argument("--no-det",      action="store_true",
                        help="Stochastic policy (shows exploration spread)")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    deterministic = not args.no_det

    if args.all_targets:
        out = args.out or "renders/rollouts_all_targets.png"
        visualize_all_targets(model, args.n_rollouts, out, deterministic)
    elif args.target:
        out = args.out or f"renders/rollouts_{args.target}.png"
        visualize_single_target(model, args.target, args.n_rollouts, out, deterministic)
    else:
        # Default: all targets grid
        out = args.out or "renders/rollouts_all_targets.png"
        visualize_all_targets(model, args.n_rollouts, out, deterministic)
