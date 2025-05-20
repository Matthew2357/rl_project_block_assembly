"""
Wrapps the AssemblyEnv in a gym interface so that it can be used with stable_baselines3
"""


import gymnasium as gym
import numpy as np
import torch
import logging

import matplotlib.pyplot as plt

# ---- project imports -------------------------------------------------------
from assembly_env import AssemblyEnv
from tasks import Bridge, Tower, DoubleBridge
from utils.logger_utils import get_logger

MAX_ACTIONS = 300 # Upper limit on the number of possible actions

class BlockAssemblyGym(gym.Env):

    def __init__(self, task, num_block_offsets: int = 1, render = False, noise=False, level = logging.INFO):
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

        self._actions = [None] * self.max_actions
        self._mask = np.zeros(self.max_actions, dtype=bool)
        self.action_space = gym.spaces.Discrete(self.max_actions)

        self.noise = noise

        h, w = self.backend.img_size
        self.observation_space = gym.spaces.Box(0, 1, shape=(h * w * 2,), dtype=np.float32) # Contains placed blocks as image and reward placement as image


    def _refresh_actions(self):
        acts = self.backend.available_actions(num_block_offsets=self.num_block_offsets)

        if not acts:
            self.logger.error("Empty action list at step %d\nBlocks: %s",
                            self.backend.step_count,
                            [b.name for b in self.backend.block_list])

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
        img_flat   = self.backend.state_feature.flatten()
        goal_flat  = self.backend.reward_feature.flatten()
        return torch.cat([img_flat, goal_flat]).numpy().astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.task = self.make_task(self.task_name,np.random.randint(1,5))
        self.backend = AssemblyEnv(self.task, level=self.logger.level)
        self.backend.reset()
        self._refresh_actions()
        obs = self._get_obs()
        return obs, {}

    def step(self, index : int):
        """
        Returns: obs, reward, terminted, truncated, info
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
        obs, reward, terminated = self.backend.step(action, self.noise)
        obs = self._get_obs()
        self._refresh_actions()

        truncated = False  

        if self.render_enabled:
            self.render()
        return obs, reward, terminated, truncated, {}

    def render(self):
        from rendering import plot_assembly_env
        
        if self.fig is None:
            self.fig, axd = plt.subplot_mosaic("ABC", figsize=(10, 4))
            self.ax_geom, self.ax_state, self.ax_reward = axd["A"], axd["B"], axd["C"]

            self.ax_geom.set_title("Assembly")
            self.ax_state.set_title("State")
            self.ax_reward.set_title("Reward")

            for ax in (self.ax_state, self.ax_reward):
                ax.set_xticks([]); ax.set_yticks([])

            # --- create the image artists with correct 2‑D shape ----------
            img0 = self.backend.state_feature.squeeze(0).numpy()   # (64,64)
            self.img_state  = self.ax_state.imshow(
                img0, cmap="gray", interpolation="none"
            )
            self.img_reward = self.ax_reward.imshow(
                self.backend.reward_feature.numpy(),
                cmap="viridis", interpolation="none"
            )

        # -------- update each frame ---------------------------------------
        self.ax_geom.clear()
        plot_assembly_env(self.backend, fig=self.fig, ax=self.ax_geom, task=self.task)

        # state feature: squeeze to 2‑D
        self.img_state.set_data(self.backend.state_feature.squeeze(0).numpy())
        self.img_reward.set_data(self.backend.reward_feature.numpy())  

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