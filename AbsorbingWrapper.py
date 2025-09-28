import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AbsorbingWrapper(gym.Wrapper):

    def __init__(self, env, horison=None):
        super().__init__(env)
        self.horison, self._elapsed_steps, self._absorbing, self._last_obs, self.reward_store = horison, 0, False, None,0

    def reset(self, **kwargs):
        self._elapsed_steps, self._absorbing, self._last_obs, self.reward_store = 0, False, None, 0
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):

        if self._absorbing:
            obs = self._last_obs
            reward = 0.0
            terminated = truncated = False
            info = {}
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._last_obs = obs
            #hide reward until horizon
            if reward > 0:
                self.reward_store=reward
                reward = 0.0
            #if env ended early then enter absorbing mode
            if terminated or truncated:
                self._absorbing = True
                terminated = truncated = False
                reward = 0.0

        self._elapsed_steps += 1

        #at horizon: return average reward, terminate
        if self.horison is not None and self._elapsed_steps >= self.horison:
            reward = self.reward_store
            terminated = True
            truncated = True
            self._absorbing = False  # reset flag

        return obs, float(reward), terminated, truncated, info
