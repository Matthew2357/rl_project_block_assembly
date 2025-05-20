"""
Modified run_policy.py to ensure visualization works properly
"""

import argparse
import logging
import time
import os
import matplotlib

# Use TkAgg backend which works better for interactive plotting
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

# --- project imports -------------------------------------------------------
from assembly_gym import BlockAssemblyGym
from utils.logger_utils import get_logger

ALGOS = {
    "maskppo": MaskablePPO,
    "ppo": PPO,
}


def main():
    args = create_parser().parse_args()

    logger = get_logger(__name__)
    level = logging.INFO

    if args.debug:
        level = logging.DEBUG
        logger.setLevel(level)
        logger.debug("Debug logging is enabled.")

    # Create the environment
    env = BlockAssemblyGym(
        task=args.task,
        render=args.render,
        level=level,
        use_obstacles=args.use_obstacles,  # Pass obstacles flag
        filter_collisions=args.filter_collisions  # Pass collision filtering flag
    )

    # Set up interactive mode for matplotlib
    if args.render:
        plt.ion()  # Turn on interactive mode
        plt.show()  # Create a figure window

    # Choose the right algorithm
    algo = ALGOS[args.algo]

    # Load the trained model
    logger.info(f"Loading model from {args.model}")
    model = algo.load(args.model)

    # Run for n_episodes
    total_rewards = []
    for episode in range(args.n_episodes):
        logger.info(f"Launching sim {episode + 1}")
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            step_count += 1
            logger.debug(f"-------------- STEP {step_count} --------------")

            # Select action
            if args.algo == "maskppo":
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            action = int(action)  # Convert from numpy int to Python int
            logger.debug(f"Selected action {action}: {env.get_action_by_index(env._actions, action)}")
            logger.debug("-------------------------------------")

            # Take action
            obs, reward, terminated, truncated, _ = env.step(action)

            # Update total reward
            total_reward += reward

            # Check if done
            done = terminated or truncated

            # Force render to ensure visualization is updated
            if args.render:
                env.render()
                plt.draw()  # Update the figure
                plt.pause(0.01)  # Small pause to allow the GUI to update

        logger.info(f"Episode {episode + 1} finished after {step_count} steps — total reward: {total_reward:.3f}")
        total_rewards.append(total_reward)

    logger.info(f"Mean total reward over {args.n_episodes} episodes: {np.mean(total_rewards):.3f}")

    # Keep the plot window open at the end if rendering
    if args.render:
        print("Press Enter to close the visualization...")
        input()  # Wait for user input before closing

    env.close()


def create_parser():
    parser = argparse.ArgumentParser(description="Run a trained policy")
    parser.add_argument("--model", required=True, help="Path to the model .zip file")
    parser.add_argument("--task", choices=["bridge", "tower", "double_bridge"], default="bridge")
    parser.add_argument("--algo", choices=list(ALGOS.keys()), default="maskppo")
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    # Add arguments for the new features
    parser.add_argument("--use-obstacles", action="store_true", help="Add obstacles to the environment")
    parser.add_argument("--filter-collisions", action="store_true",
                        help="Filter out actions that would cause collisions")
    return parser


if __name__ == "__main__":
    main()