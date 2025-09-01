import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.core.mission import MissionSpace

class MazeEnv(MiniGridEnv):
    def __init__(self,size=10,max_steps=100,extra_connections=0,extra_connection_prob=None,random_goal=False,**kwargs):
        mission_space = MissionSpace(mission_func=lambda: "reach the green goal")
        super().__init__(mission_space=mission_space,grid_size=size,max_steps=max_steps,see_through_walls=True,**kwargs)
        self.extra_connections = int(extra_connections)
        self.extra_connection_prob = extra_connection_prob
        self.random_goal = bool(random_goal)
        self.goal_pos = None

    #generates the maze, places the agent and the goal
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        W, H = width, height

        #Binary maze: 1=wall, 0=passage
        maze = np.ones((W, H), dtype=np.int8)  #fill entire maze with walls

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

        #OPTIONAL: Add extra connections
        #======================================================================#
        candidates = []
        for x in range(1, W - 1):
            for y in range(1, H - 1):
                if maze[x, y] != 1:
                    continue
                horiz = (maze[x - 1, y] == 0 and maze[x + 1, y] == 0 and maze[x, y - 1] == 1 and maze[x, y + 1] == 1)
                vert  = (maze[x, y - 1] == 0 and maze[x, y + 1] == 0 and maze[x - 1, y] == 1 and maze[x + 1, y] == 1)
                if horiz or vert:
                    candidates.append((x, y))
        #either use the extra_connection_prob or the extra_connections method
        if self.extra_connection_prob is not None:
            p = float(self.extra_connection_prob)
            for (x, y) in candidates:
                if rng.random() < p:
                    maze[x, y] = 0
        elif self.extra_connections > 0 and len(candidates) > 0:
            k = min(int(self.extra_connections), len(candidates))
            idxs = rng.choice(len(candidates), size=k, replace=False)
            for i in (idxs if np.iterable(idxs) else [idxs]):
                x, y = candidates[int(i)]
                maze[x, y] = 0
        #======================================================================#

        #write maze into minigrid
        for x in range(1, W - 1):
            for y in range(1, H - 1):
                self.grid.set(x, y, None if maze[x, y] == 0 else Wall())

        #agent placement
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        #goal placement (uniform if random_goal, else fixed corner)
        if self.random_goal:
            free = [(x, y) for x in range(1, W - 1) for y in range(1, H - 1)
                    if maze[x, y] == 0 and (x, y) != self.agent_pos]
            if free:
                idx = int(rng.integers(len(free)))
                gx, gy = free[idx]
            else:
                gx, gy = (W - 2, H - 2)
        else:
            gx, gy = (W - 2, H - 2)

        self.goal_pos = (gx, gy)
        self.put_obj(Goal(), gx, gy)
        self.mission = "reach the green goal"
