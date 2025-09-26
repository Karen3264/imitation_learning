#A*
from heapq import heappush, heappop
from minigrid.core.actions import Actions
from minigrid.core.world_object import Goal, Door
import random

def _is_passable_cell(env, x, y):
    #stay inside the interior frame
    if x <= 0 or y <= 0 or x >= env.width - 1 or y >= env.height - 1:
        return False
    obj = env.grid.get(x, y)
    if obj is None: return True
    return getattr(obj, "can_overlap", lambda: False)() #for goal

def _heuristic(x, y, gx, gy, curr_dir):
    #heuristic = Manhattan distance (min #forward steps ignoring heading)+ an optimistic turn estimate (at most 1 turn).
    dx, dy = gx - x, gy - y
    manhattan = abs(dx) + abs(dy)
    if manhattan == 0: return 0
    desired = []
    if dx: desired.append(0 if dx > 0 else 2) #axis: 0 (→) or 2 (←)
    if dy: desired.append(1 if dy > 0 else 3) #y-axis: 1 (↓) or 3 (↑)
    td = min((((h - curr_dir) % 4) for h in desired), default=0)
    td = min(td, 4 - td)  #minimal turns to face a helpful axis
    return manhattan + (1 if td > 0 else 0)
def astar_plan(env, agent_pos=None, goal_pos=None):
    (ax, ay), (gx, gy), d0 = agent_pos, goal_pos, int(env.agent_dir)
    start = (ax, ay, d0)
    g, parent, act_from_parent = {start: 0}, {start: None}, {start: None}
    queue, closed = [], set()
    heappush(queue, (0, 0, random.random(), start)) #push candidate state with priority f into the queue

    goal_state = None
    while len(queue) > 0: #keep looping as long as there are elements in the queue
        _, _, _, s = heappop(queue) #pop the state with the best priority (lowest f) to expand next
        if s in closed:
            continue
        closed.add(s)
        x, y, d = s
        if (x, y) == (gx, gy):
            goal_state = s
            break
        #=====Expand 3 successors from (x,y,d): turn left, turn right, move forward=====#
        #LEFT
        sL = (x,y,(d - 1)%4) #new state
        if g[s] + 1 < g.get(sL, 1e9): #does the path “current best cost to s” + cost of this action (1) beat the best cost we’ve seen for sL, if unseen 1e9 is large
            g[sL] = g[s] + 1 #update the best known cost-to-reach sL
            parent[sL] = s #backpointer
            act_from_parent[sL] = Actions.left #backpointer
            f = g[sL] + _heuristic(x, y, gx, gy, sL[2]) #compute the A* priority
            heappush(queue, (f, g[sL], random.random(), sL)) 
        #RIGHT
        sR = (x,y,(d + 1)%4) #new state
        if g[s] + 1 < g.get(sR, 1e9):
            g[sR] = g[s] + 1
            parent[sR] = s
            act_from_parent[sR] = Actions.right
            f = g[sR] + _heuristic(x, y, gx, gy, sR[2])
            heappush(queue, (f, g[sR], random.random(), sR))
        #FORWARD
        DX = (1, 0, -1, 0)  #0=right,1=down,2=left,3=up
        DY = (0, 1, 0, -1)
        nx, ny = x + DX[d], y + DY[d] #new state x, y, d remains same TO SEE IN WHAT DIRECTION we're going
        if _is_passable_cell(env, nx, ny):
            sF = (nx, ny, d)
            if g[s] + 1 < g.get(sF, 1e9):
                g[sF] = g[s] + 1
                parent[sF] = s
                act_from_parent[sF] = Actions.forward
                f = g[sF] + _heuristic(nx, ny, gx, gy, d) #heuristic now evaluated at the new cell (nx, ny) with same dir d
                heappush(queue, (f, g[sF], random.random(), sF))

    if goal_state is None:
        return {}  # no path found

    #===== backtrack path and build lookup =====#
    path_states, path_actions = [], []
    s = goal_state
    while parent[s] is not None:
        path_states.append(s)
        path_actions.append(act_from_parent[s])
        s = parent[s]
    path_states.reverse()
    path_actions.reverse()
    #return a dictionary 
    lookup = {}
    s = goal_state
    while parent[s] is not None:
        p = parent[s]
        a = act_from_parent[s]
        key = f"{p[0]}_{p[1]}_{p[2]}"
        lookup[key] = int(a)
        s = p
    return lookup

import torch
from stable_baselines3.common.policies import BasePolicy
from imitation.policies.serialize import policy_registry
import numpy as np
class AStarPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, env=None, calc_path_each_step=True):
        super().__init__(observation_space, action_space)
        self.env = env
        self.cached_paths = {}
        self.calc_path_each_step=calc_path_each_step


    def forward(self, obs, deterministic=False):
        return None

    def _predict(self, observation, deterministic=False):
        acts = []
        if hasattr(self.env, "envs"):
            for i, base_env in enumerate(self.env.envs):
                base_env = base_env.unwrapped
                state=str(base_env.agent_pos[0])+"_"+str(base_env.agent_pos[1])+"_"+str(int(base_env.agent_dir))
                if i not in self.cached_paths or state not in self.cached_paths[i]:
                    self.cached_paths[i] = astar_plan(base_env, agent_pos=tuple(base_env.agent_pos),goal_pos=getattr(base_env, "goal_pos", (9, 9)))
                act = self.cached_paths[i].get(state, int(Actions.done))
                acts.append(act)
        else:
            base_env = self.env.unwrapped
            state=str(base_env.agent_pos[0])+"_"+str(base_env.agent_pos[1])+"_"+str(int(base_env.agent_dir))
            if 0 not in self.cached_paths or state not in self.cached_paths[0]:
                self.cached_paths[0] = astar_plan(base_env, agent_pos=tuple(base_env.agent_pos), goal_pos=getattr(base_env, "goal_pos", (9, 9)))
            act = self.cached_paths[0].get(state, int(Actions.done))
            acts.append(act)

        if self.calc_path_each_step: #useful when dynamic or noisy environment
          self.cached_paths = {}


        return torch.as_tensor(acts, device=self.device)


