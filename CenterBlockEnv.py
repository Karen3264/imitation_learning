import numpy as np
import gymnasium as gym
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Goal, Wall

class CenterBlockEnv(EmptyEnv):
    def __init__(self, size=11, block_size=3, **kwargs):
        super().__init__(size=size, **kwargs)
        self.block_size = block_size
        self.goal_pos = None

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        midx = width // 2
        start = (midx, 1)
        goal  = (midx, height - 2)
        self.agent_pos = np.array(start, dtype=int)
        self.agent_dir = 1
        #centered wall block
        cx, cy = width // 2, height // 2
        r = self.block_size // 2
        for x in range(cx - r, cx + r + 1):
            for y in range(cy - r, cy + r + 1):
                self.grid.set(x, y, Wall())
        #goal
        self.put_obj(Goal(), *goal)
        self.goal_pos = goal
