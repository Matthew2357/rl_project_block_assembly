#!/usr/bin/env python3
"""
Run a trained block‑assembly policy.

Usage example
-------------
python run_policy.py --model runs/bridge_maskppo_20250506_113000/final_model.zip \
                     --task bridge --algo maskppo --render
"""

import argparse
import logging
import time
import numpy as np
from pathlib import Path
import imageio.v2 as imageio

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ------------------------------------------------------------------------
# project imports (same as train.py)
# ------------------------------------------------------------------------
from assembly_gym import BlockAssemblyGym
from utils.logger_utils import get_logger

ALGOS = {
    "ppo": PPO,
    "maskppo": MaskablePPO,
}

def _capture_figure(fig):
    fig.canvas.draw()                                    
    rgba = np.asarray(fig.canvas.buffer_rgba())          
    return rgba[:, :, :3].copy()                         


def _log_step_debug(logger, base_env, action_idx, step):
    available = base_env.backend.available_actions()
    a = base_env.get_action_by_index(available, action_idx)
    logger.debug("-------------- STEP %d --------------", step)
    logger.debug("Selected action %s: %s", action_idx, a)
    logger.debug("-------------------------------------")

# ------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------
def main():
    args = parse_args()

    # Logger setup
    logger = get_logger(__name__)
    level = logging.INFO

    if args.debug:
        level = logging.DEBUG
        logger.setLevel(level)
        logger.debug("Debug logging is enabled.")

    # Create environment
    base_env = BlockAssemblyGym(task=args.task,
                                level=level,
                                noise = args.noise,
                                render=args.render)

    if args.algo == "maskppo":
        env = ActionMasker(base_env, lambda e: e.get_action_mask())
    else:
        env = base_env

    # Load SB3 model
    Algo = ALGOS[args.algo]
    logger.info(f"Loading model from {args.model}")
    model = Algo.load(args.model, device=args.device)

    # GIF recording
    record_gif: bool = bool(args.gif)
    gif_path: Path | None = (
        Path("Images") / args.gif if isinstance(args.gif, str)          
        else Path("Images/rollout.gif")                    
    ) if record_gif else None
    frames: list[np.ndarray] = []

    # Rollout loop
    for episode_idx in range(1, args.n + 1):
        model = Algo.load(args.model, device=args.device)
        obs, _ = env.reset(seed=args.seed)
        done, truncated = False, False
        total_reward = 0.0
        steps = 0
        logger.info(f"Launching sim {episode_idx}")

        # Rollout one ep
        while not (done or truncated):
            if args.algo == "maskppo":
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=False, action_masks=action_masks)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, _info = env.step(action)
            total_reward += reward
            steps += 1
            
            if record_gif and episode_idx == 1 and args.render:
                frames.append(_capture_figure(base_env.fig))

            if args.debug:
                _log_step_debug(logger, base_env, action, steps)
            
            time.sleep(0.1)
        
        logger.info(f"Episode {episode_idx} finished after {steps} steps — total reward: {total_reward:.3f}")

        if record_gif and frames:
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(gif_path, frames, fps=3, loop=0)
            logger.info(f"GIF saved to {gif_path} ({len(frames)} frames)")

# ------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run a trained block‑assembly policy")
    p.add_argument("--model", required=True, help="Path to saved .zip model")
    p.add_argument("--task", choices=["bridge", "tower", "double_bridge"], default="bridge")
    p.add_argument("--num-stories", type=int, default=2)
    p.add_argument("--noise", action="store_true")
    p.add_argument("--algo", choices=list(ALGOS.keys()), default="maskppo")
    p.add_argument("--device", default="auto")
    p.add_argument("--render", action="store_true", help="Render environment while running")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("-n", type=int, default=4, help="Number of times to run the policy")
    p.add_argument("--gif", nargs="?",const=True,help="Save GIF of the first rollout (optionally pass a filename)")
    return p.parse_args()

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
