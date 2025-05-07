import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import assembly_env as ba
import numpy as np

from tasks import Bridge
import torch

from sb3_contrib.common.wrappers import ActionMasker


register(
    id="block-assembly-env-v0",
    entry_point="gym_wrapper:BlockAssemblyGym",
)

class BlockAssemblyGym(gym.Env):
    #metadata = {"render_modes":[None], "render_fps":1}

    def __init__(self, task):

        super().__init__()
        self.env = ba.AssemblyEnv(task)
        
        self.actions = self.env.available_actions()
        self.action_space = spaces.Discrete(
            len(self.actions)
            )
        self.action_mapping = self.create_action_mapping()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.env.img_size, 
            dtype=np.float32
        )

    def create_action_mapping(self):
        # Create a mapping from discrete action indices to the actual Action objects
        actions = self.env.available_actions()
        return {i: action for i, action in enumerate(actions)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()

        self.env.state_feature = torch.zeros(self.env.img_size)  # Clear state

        obs = self.env.state_feature.numpy().astype(np.float32)
        info = {}

        

        return obs, info

    def step(self, action):
        obs, reward, terminated = self.env.step(self.action_mapping[action])
        info = {}
        self.actions = self.env.available_actions()
        self.action_space = spaces.Discrete(
            len(self.actions)
            )
        return obs.numpy(), reward, terminated, False, info
    
    def render(self):
        pass

    def valid_action_mask(self):
        mask = np.zeros(len(self.actions), dtype=bool)
        current_actions = self.actions

        # Match by hashing or object equality
        valid_hashes = set(hash(a) for a in current_actions)
        for idx, a in enumerate(self.all_actions):
            if hash(a) in valid_hashes:
                mask[idx] = True
        return mask



if __name__ == '__main__':
    env = gym.make("block-assembly-env-v0", task=Bridge(num_stories=2))
    #env = ActionMasker(env, lambda env: env.valid_action_mask())
    check_env(env.unwrapped)