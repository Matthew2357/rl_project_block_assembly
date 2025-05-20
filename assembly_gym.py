"""
Wrapps the AssemblyEnv in a gym interface so that it can be used with stable_baselines3
With added obstacles and collision filtering
"""

import gymnasium as gym
import numpy as np
import torch
import logging
import random

import matplotlib.pyplot as plt

# ---- project imports -------------------------------------------------------
from assembly_env import AssemblyEnv
from tasks import Bridge, Tower, DoubleBridge
from utils.logger_utils import get_logger

MAX_ACTIONS = 300  # Upper limit on the number of possible actions

class BlockAssemblyGym(gym.Env):

    def __init__(self, task, num_block_offsets: int = 1, render = False, level = logging.INFO,
                 use_obstacles=True, filter_collisions=True):
        super().__init__()
        self.logger = get_logger(__name__)
        self.logger.setLevel(level)

        self.task_name = task
        self.task = self.make_task(self.task_name, 2)
        self.backend = AssemblyEnv(self.task,level=level)

        self.num_block_offsets = num_block_offsets
        self.max_actions = MAX_ACTIONS
        self.render_enabled = render
        self.fig, self.ax = None, None

        # New flags for the added features
        self.use_obstacles = use_obstacles
        self.filter_collisions = filter_collisions

        # Generate obstacles when initializing
        self.obstacles = []
        if self.use_obstacles:
            self._generate_obstacles()

        self._actions = [None] * self.max_actions
        self._mask = np.zeros(self.max_actions, dtype=bool)
        self.action_space = gym.spaces.Discrete(self.max_actions)

        h, w = self.backend.img_size
        # Adding an extra channel for obstacles if enabled
        if self.use_obstacles:
            self.observation_space = gym.spaces.Box(0, 1, shape=(h * w * 3,), dtype=np.float32)  # Contains placed blocks, reward placement, and obstacles
        else:
            self.observation_space = gym.spaces.Box(0, 1, shape=(h * w * 2,), dtype=np.float32)  # Contains placed blocks and reward placement

    def _generate_obstacles(self):
        """Generate random obstacles in the environment"""
        h, w = self.backend.img_size
        num_obstacles = random.randint(2, 5)  # Random number of obstacles
        self.obstacles = []

        # Create obstacle representation - each obstacle is (x, y, width, height)
        for _ in range(num_obstacles):
            x = random.randint(0, w-5)
            y = random.randint(0, h//2)  # Keep obstacles in lower half to not interfere too much with task
            width = random.randint(3, 8)
            height = random.randint(3, 8)
            self.obstacles.append((x, y, width, height))

    def _is_colliding_with_obstacle(self, action):
        """Check if the proposed action would collide with any obstacle"""
        if not self.obstacles:
            return False

        # This is a simplified collision detection - you'll need to adapt this
        # to match your actual block and action representation

        # Extract action details - assuming action contains placement coordinates
        # This needs to be adapted based on your actual action representation
        target_block = action.target_block
        target_face = action.target_face
        shape = action.shape

        # Get the actual position where the block would be placed
        # This is just a placeholder - you need to implement the actual collision detection
        # based on your environment's geometry

        # Assuming we can get the position from the action or by simulating it
        # This is a simplified example
        proposed_position = self._get_position_from_action(action)

        if proposed_position is None:
            return False

        x, y, width, height = proposed_position

        # Check collision with any obstacle
        for obs_x, obs_y, obs_width, obs_height in self.obstacles:
            # Simple rectangle collision detection
            if (x < obs_x + obs_width and
                x + width > obs_x and
                y < obs_y + obs_height and
                y + height > obs_y):
                return True

        return False

    def _get_position_from_action(self, action):
        """
        Convert an action to a position in the environment
        This is a placeholder - implement based on your actual environment
        """
        # This is where you need to implement the logic to convert an action
        # to a position (x, y, width, height) in your environment
        # For example, you might use the backend to simulate the action
        # and get the resulting position

        # Placeholder implementation
        try:
            # You'll need to implement this based on your actual environment
            # For example, you might use:
            # - action.target_block.position
            # - Calculate based on target_face and offset
            # - Use backend internals to get position information

            # For now, we'll use a dummy implementation that returns None
            # which means no collision
            return None
        except:
            return None

    def _create_obstacle_feature(self):
        """Create a feature map representing obstacles"""
        h, w = self.backend.img_size
        obstacle_feature = np.zeros((h, w), dtype=np.float32)

        # Draw obstacles on the feature map
        for x, y, width, height in self.obstacles:
            obstacle_feature[y:y+height, x:x+width] = 1.0

        return torch.tensor(obstacle_feature, dtype=torch.float32)

    def _refresh_actions(self):
        acts = self.backend.available_actions(num_block_offsets=self.num_block_offsets)

        if not acts:
            self.logger.error("Empty action list at step %d\nBlocks: %s",
                            self.backend.step_count,
                            [b.name for b in self.backend.block_list])

        # Filter out actions that would cause collision with obstacles
        if self.filter_collisions and self.obstacles:
            filtered_acts = []
            for act in acts:
                if not self._is_colliding_with_obstacle(act):
                    filtered_acts.append(act)

            # If all actions would collide, we keep the original set
            # to prevent the agent from having no valid actions
            if filtered_acts:
                acts = filtered_acts
            else:
                self.logger.warning("All actions would collide with obstacles, keeping original set")

        if len(acts) > self.max_actions:
            raise ValueError(f"Too many actions ({len(acts)}) for max_actions ({self.max_actions}).")

        # pad to fixed size
        self._actions = acts + [None] * (self.max_actions - len(acts))
        self._mask[: len(acts)] = True
        self._mask[len(acts) :] = False
        self._mask[self.max_actions - 1] = True

    def action_masks(self):
        return self.get_action_mask()

    def get_action_mask(self):
        if not self._mask.any() :
            self.logger.debug("Empty Mask")
        return self._mask

    def _get_obs(self):
        img_flat = self.backend.state_feature.flatten()
        goal_flat = self.backend.reward_feature.flatten()

        if self.use_obstacles:
            obstacle_flat = self._create_obstacle_feature().flatten()
            return torch.cat([img_flat, goal_flat, obstacle_flat]).numpy().astype(np.float32)
        else:
            return torch.cat([img_flat, goal_flat]).numpy().astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.task = self.make_task(self.task_name, np.random.randint(1, 5))
        self.backend = AssemblyEnv(self.task, level=self.logger.level)
        self.backend.reset()

        # Generate new obstacles on reset
        if self.use_obstacles:
            self._generate_obstacles()

        self._refresh_actions()
        obs = self._get_obs()
        return obs, {}

    def step(self, index: int):
        """
        Returns: obs, reward, terminated, truncated, info
        """

        if index == (self.max_actions - 1):
            self.logger.debug("Noop chosen - ending episode")
            obs = self._get_obs()
            self._refresh_actions()
            return obs, 0.0, True, False, {}   # terminate with zero reward

        if not self._mask[index]:
            obs = self._get_obs()
            self._refresh_actions()
            self.logger.warning("Invalid action choice")
            return obs, -1.0, False, False, {}

        if self._actions is None:
            obs = self._get_obs()
            self.logger.warning("No action available")
            return obs, 0.0, True, False, {}

        action = self._actions[index]
        obs, reward, terminated = self.backend.step(action)

        # Apply penalty if the action collides with an obstacle
        # This is an additional mechanism beyond filtering actions
        if self.use_obstacles and self._is_colliding_with_obstacle(action):
            reward -= 0.5  # Penalty for collision

        obs = self._get_obs()
        self._refresh_actions()

        truncated = False

        if self.render_enabled:
            self.render()
        return obs, reward, terminated, truncated, {}

    def render(self):
        from rendering import plot_assembly_env

        if self.fig is None:
            if self.use_obstacles:
                self.fig, axd = plt.subplot_mosaic("ABCD", figsize=(15, 4))
                self.ax_geom, self.ax_state, self.ax_reward, self.ax_obstacles = axd["A"], axd["B"], axd["C"], axd["D"]
                self.ax_obstacles.set_title("Obstacles")
                self.ax_obstacles.set_xticks([]); self.ax_obstacles.set_yticks([])
            else:
                self.fig, axd = plt.subplot_mosaic("ABC", figsize=(10, 4))
                self.ax_geom, self.ax_state, self.ax_reward = axd["A"], axd["B"], axd["C"]

            self.ax_geom.set_title("Assembly")
            self.ax_state.set_title("State")
            self.ax_reward.set_title("Reward")

            for ax in (self.ax_state, self.ax_reward):
                ax.set_xticks([]); ax.set_yticks([])

            # --- create the image artists with correct 2‑D shape ----------
            img0 = self.backend.state_feature.squeeze(0).numpy()   # (64,64)
            self.img_state = self.ax_state.imshow(
                img0, cmap="gray", interpolation="none"
            )
            self.img_reward = self.ax_reward.imshow(
                self.backend.reward_feature.numpy(),
                cmap="viridis", interpolation="none"
            )

            if self.use_obstacles:
                self.img_obstacles = self.ax_obstacles.imshow(
                    self._create_obstacle_feature().numpy(),
                    cmap="Reds", interpolation="none"
                )

        # -------- update each frame ---------------------------------------
        self.ax_geom.clear()
        plot_assembly_env(self.backend, fig=self.fig, ax=self.ax_geom, task=self.task)

        # state feature: squeeze to 2‑D
        self.img_state.set_data(self.backend.state_feature.squeeze(0).numpy())
        self.img_reward.set_data(self.backend.reward_feature.numpy())

        if self.use_obstacles:
            self.img_obstacles.set_data(self._create_obstacle_feature().numpy())

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        pass

    def make_task(self, name: str, num_stories: int):
        if name == "bridge":
            return Bridge(num_stories=num_stories)
        if name == "tower":
            # tower has a list of (x, height) targets – we centre at 0
            targets = [(0, h) for h in range(1, num_stories + 1)]
            return Tower(targets)
        if name == "double_bridge":
            return DoubleBridge(num_stories=num_stories)
        raise ValueError(f"Unknown task '{name}'.")
    
    def print_available_actions(self):
        """Prints all available actions one per line with index."""
        actions = self.backend.available_actions()
        for i, a in enumerate(actions):
            print(f"[{i:03}] target_block={a.target_block}, "
                f"target_face={a.target_face}, shape={a.shape}, "
                f"face={a.face}, offset_x={a.offset_x}")
        
    def get_action_by_index(self, actions, index):
        """Returns the action at the given index, or raises IndexError."""
        if 0 <= index < len(actions):
            return actions[index]
        raise IndexError(f"Index {index} out of bounds (len={len(actions)})")