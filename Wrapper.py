import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyWrapper(gym.Wrapper):
    def __init__(self, env, horizon=None):
        super().__init__(env)
        self.horizon = horizon
        self._elapsed_steps = 0
        self._padding = False
        self.true_terminate=False
        self._last_obs = None
        #ACTION SPACE
        self.action_space = spaces.Discrete(3)
        self._action_map = {0: self.unwrapped.actions.left,1: self.unwrapped.actions.right,2: self.unwrapped.actions.forward,}
        #OBSERVATION SPACE
        H, W, _ = self.observation_space["image"].shape
        self.agent_view_size = getattr(env.unwrapped, "agent_view_size", None)
        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, 1), dtype=np.uint8)
        #REWARD: Store and only return when episode has ended
        self.reward_store = []

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self._padding = False
        self.true_terminate=False
        self.reward_store = []
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        action = self._action_map[action] #map restricted action to MiniGrid action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.horizon == None:
          return self._process_obs(obs), float(reward), terminated, truncated, info

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
        if self._elapsed_steps >= self.horizon:
          arr = np.array(self.reward_store)
          reward = np.mean(arr) if arr.size > 0 else 0.0
          terminated = self.true_terminate
          truncated = True

        return self._process_obs(obs), float(reward), terminated, truncated, info


    #MAKE OBSERVATION NICE IN PARTIALLY AND FULLY OBSERVED CASE
    def _process_obs(self, obs):
        grid = obs["image"][:, :, 0:1]
        return grid




    
