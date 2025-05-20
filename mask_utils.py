"""
Helper function to fix mask-related issues in MaskablePPO
Add this to a utils file or directly import in your code
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional, Any, Tuple


def fix_mask_for_sb3(
        mask: np.ndarray,
        epsilon: float = 1e-8
) -> np.ndarray:
    """
    Ensures that action masks meet the constraints required by SB3's MaskablePPO.

    The action mask must have at least one True value to avoid distribution errors.
    This is particularly important for the Categorical distribution used in Stable Baselines 3.

    Args:
        mask: Boolean numpy array representing the action mask (True for valid actions)
        epsilon: Small value to ensure numerical stability

    Returns:
        Modified action mask with at least one valid action
    """
    # Make a copy to avoid modifying the original
    fixed_mask = mask.copy()

    # Ensure at least one action is available
    if not fixed_mask.any():
        # If no valid actions, enable the last action (typically NOOP)
        fixed_mask[-1] = True

    return fixed_mask


# Example of a patched wrapper for ActionMasker
class SafeActionMasker:
    """
    A decorator class for env.action_masks() to ensure valid probability distributions
    when using MaskablePPO.

    Usage:
    env = ActionMasker(env, SafeActionMasker.action_mask_fn)

    where action_mask_fn is a function that calls env.action_masks() and applies fix_mask_for_sb3
    """

    @staticmethod
    def action_mask_fn(env) -> np.ndarray:
        """
        Returns a valid action mask for the environment

        Args:
            env: Gym environment with action_masks method

        Returns:
            Valid action mask for the environment
        """
        masks = env.action_masks()
        return fix_mask_for_sb3(masks)