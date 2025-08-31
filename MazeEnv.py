import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.core.mission import MissionSpace

class MazeEnv(MiniGridEnv):

    def __init__(self, size=10, max_steps=None, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "reach the green goal")
        if max_steps is None:
            max_steps = 4 * size * size
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps,see_through_walls=True, **kwargs,)

    #generates the maze, places the agent and the goal
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height) #make grid
        self.grid.wall_rect(0, 0, width, height) #make walls

        #Binary maze: 1=wall, 0=passage
        W, H = width, height
        maze = np.ones((W, H), dtype=np.int8) #fill entire maze with walls
  
 
        rng = self.np_random
        stack = [(1, 1)] 
        maze[1, 1] = 0 #agent start pos is walkable

        def neighbors(x, y):
            dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)] #use 2 to move over wall
            rng.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < W - 1 and 1 <= ny < H - 1 and maze[nx, ny] == 1:
                    yield nx, ny, dx, dy

        while stack:
            x, y = stack[-1]
            nbrs = list(neighbors(x, y))
            if not nbrs:
                stack.pop()
                continue
            nx, ny, dx, dy = nbrs[0]
            maze[x + dx // 2, y + dy // 2] = 0 #knock down wall
            maze[nx, ny] = 0
            stack.append((nx, ny))

        #write maze into minigrid
        for x in range(1, W - 1):
            for y in range(1, H - 1):
                self.grid.set(x, y, None if maze[x, y] == 0 else Wall())

        #agent & goal
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        gx, gy = (W-2, H-2)  #goal position
        self.put_obj(Goal(), gx, gy) #put down the goal 
        self.mission = "reach the green goal"
