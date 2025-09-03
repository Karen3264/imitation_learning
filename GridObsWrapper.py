from gymnasium import ObservationWrapper
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from minigrid.wrappers import ImgObsWrapper

class GridObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = ImgObsWrapper(env)
        h, w, _ = self.env.observation_space.shape
        self.observation_space = Box(low=0, high=2, shape=(h, w), dtype=np.uint8) #must be set for API
    def observation(self, obs):
        obj = obs[:, :, 0]  #take object channel
        simple = np.zeros_like(obj, dtype=np.int64)
        simple[obj == 1] = 0  #empty
        simple[obj == 2] = 1  #wall
        simple[obj == 8] = 2  #goal
        return simple 
