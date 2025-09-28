import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from minigrid.envs.empty import EmptyEnv
from minigrid.core.world_object import Goal
from minigrid.core.actions import Actions


class RandomGoalEmptyEnv(EmptyEnv):
    def __init__(self, size=11, deterministic=False, start_pos=None, random_start=False, **kwargs):
        super().__init__(size=size, **kwargs)
        self.goal_pos = None
        self.deterministic = deterministic
        self.start_pos = tuple(start_pos) if start_pos is not None else None
        self.random_start = random_start

        # Restrict action space: 0=left, 1=right, 2=forward
        self.action_space = Discrete(3)

    def step(self, action):
        # Map reduced action space → full MiniGrid action space
        if action == 0:
            real_action = Actions.left
        elif action == 1:
            real_action = Actions.right
        elif action == 2:
            real_action = Actions.forward
        else:
            raise ValueError(f"Invalid action {action}")

        obs, reward, terminated, truncated, info = super().step(real_action)
        return obs, float(reward), terminated, truncated, info

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)  # generate the base empty grid
        self.grid.set(width - 2, height - 2, None)  # remove default goal

        if self.start_pos is not None:
            self.agent_pos = self.start_pos

        rng = self.np_random
        if self.random_start:
            x = rng.integers(1, width - 1)
            y = rng.integers(1, height - 1)
            self.agent_pos = (x, y)
            self.agent_dir = rng.integers(0, 4)

        if self.deterministic:
            x, y = width - 2, height - 2
        else:
            while True:
                x = rng.integers(1, width - 1)
                y = rng.integers(1, height - 1)
                if (x, y) != tuple(self.agent_pos):
                    break

        self.put_obj(Goal(), x, y)
        self.goal_pos = (x, y)
        self.mission = "get to the green goal square"
