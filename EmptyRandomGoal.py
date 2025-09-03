import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import ImgObsWrapper
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Goal


class RandomGoalEmptyEnv(EmptyEnv):
    def __init__(self, size=11, deterministic=False, **kwargs):
        super().__init__(size=size, **kwargs)
        self.goal_pos = None
        self.deterministic = deterministic

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height) #generate the base empty grid
        #remove the default goal
        self.grid.set(width - 2, height - 2, None)

        if self.deterministic:
            x, y = width - 2, height - 2
        else:
        #sample somewhere for goal to be
            rng = self.np_random
            while True:
                x = rng.integers(1, width - 1)
                y = rng.integers(1, height - 1)
                if (x, y) != tuple(self.agent_pos):
                    break
        self.put_obj(Goal(), x, y) #place goal
        self.goal_pos = (x,y) 
