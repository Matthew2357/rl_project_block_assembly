import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import assembly_env as ba
import numpy as np

from tasks import Bridge
import torch

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO


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
        self.all_actions = self.env.all_actions()
        self.action_space = spaces.Discrete(
            len(self.all_actions)
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
        actions = self.all_actions
        return {i: action for i, action in enumerate(actions)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()

        self.env.state_feature = torch.zeros(self.env.img_size)  # Clear state

        obs = self.env.state_feature.numpy().astype(np.float32)
        info = {}
        self.actions = self.env.available_actions()

        

        return obs, info

    def step(self, action):
        obs, reward, terminated = self.env.step(self.action_mapping[action])
        info = {}
        self.actions = self.env.available_actions()
        '''self.actions = self.env.available_actions()
        self.action_space = spaces.Discrete(
            len(self.actions)
            )'''
        return obs.numpy(), float(reward), terminated, False, info
    
    def render(self):
        pass

    def valid_action_mask(self):
        mask = np.zeros(len(self.all_actions), dtype=bool)

        available_set = set(self.actions)
        all_set = set(self.all_actions)

        # Sanity check
        missing = available_set - all_set
        assert not missing, f"Invalid available actions: {missing}"

        for i, action in enumerate(self.all_actions):
            if action in available_set:
                mask[i] = True

        assert np.any(mask), "No valid actions in the current mask!"
        return mask
    
def get_action_mask(env):
    mask = env.unwrapped.valid_action_mask()
    
    # Ensure that the valid actions' probabilities are normalized
    valid_actions = np.nonzero(mask)[0]
    
    if len(valid_actions) > 0:
        # Assign equal probability to each valid action
        probs = np.zeros(len(mask))
        probs[valid_actions] = 1.0 / len(valid_actions)
    else:
        # If there are no valid actions, return a uniform distribution (should be rare or invalid)
        probs = np.ones(len(mask)) / len(mask)

    return probs


def filter_true_elements(bool_list, value_list):
    return [value_list[i] for i in range(len(bool_list)) if bool_list[i]]





if __name__ == '__main__':
    env = gym.make("block-assembly-env-v0", task=Bridge(num_stories=2))
    #Maskedenv = ActionMasker(env, lambda env: env.valid_action_mask())
    maskedEnv = ActionMasker(env, get_action_mask)
    #print(filter_true_elements(get_action_mask(env), env.unwrapped.all_actions))
    
    model = MaskablePPO(MaskableActorCriticPolicy, maskedEnv, verbose=1, n_steps=100)
    
    print("Learning...")
    model.learn(total_timesteps=10)
    print("Learning done!")
    mask = env.unwrapped.get_action_mask()
    print(len(mask))
    print(sum(mask))
    print(len(env.unwrapped.actions))
    print(len(env.unwrapped.all_actions))