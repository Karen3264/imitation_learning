import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HorisonWrapper(gym.Wrapper):
    def __init__(self, env, horison=None):
        super().__init__(env)
        self.horison = horison
        self._elapsed_steps = 0
        self._padding = False
        self.true_terminate=False
        self._last_obs = None
        self.agent_view_size = getattr(env.unwrapped, "agent_view_size", None)
        self.reward_store = []
        #ACTION SPACE
        self.action_space = spaces.Discrete(3)
        self._action_map = {0: self.unwrapped.actions.left,1: self.unwrapped.actions.right,2: self.unwrapped.actions.forward,}
        
        

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self._padding = False
        self.true_terminate=False
        self.reward_store = []
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        action = self._action_map[action] #map restricted action to MiniGrid action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.horison == None:
          return obs, float(reward), terminated, truncated, info

        if reward > 0:
          self.reward_store.append(reward)
          reward=0.0

        #if episode ended before horison reached, reset the env so that we can gather more (s,a) until total reached
        if self._padding:
            obs, info = self.env.reset()
            self._padding, terminated = False, False

        if terminated:
            self._padding, terminated, self.true_terminate = True, False, True

        self._elapsed_steps += 1
        #if the horison has been reached
        if self._elapsed_steps >= self.horison:
          arr = np.array(self.reward_store)
          reward = np.mean(arr) if arr.size > 0 else 0.0
          terminated = self.true_terminate
          truncated = True

        return obs, float(reward), terminated, truncated, info


    
