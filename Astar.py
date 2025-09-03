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

def find_optimal_action(env, agent_pos=None, goal_pos=None):
    (ax, ay), (gx, gy), d0= agent_pos, goal_pos, int(env.agent_dir)
    if (ax, ay) == (gx, gy): return int(Actions.done) #if agent pos = goal pos
    start = (ax, ay, d0) #agent pos and initial direction
    g = {start: 0} #g: best-known cost-to-reach for each explored state
    parent = {start: None} #parent: back-pointer graph to reconstruct the path (state -> parent state)
    act_from_parent = {start: None} #act_from_parent: the action taken to move from parent[state] → state
    pq = [] #Priority queue of states.
    heappush(pq, (0, 0, random.random(), start))  #(f, g, tie, state)
    closed = set()  #closed set of states whose optimal g-cost has been finalized.

    #===========A* main loop==========#
    while pq:
        #pop the most promising state (lowest f; random tie-break on equal f,g)
        _, _, _, s = heappop(pq)
        #skip if we have already finalized this state
        if s in closed:
            continue
        closed.add(s)
        #unpack the state into position and heading
        x, y, d = s

        #=====if at goal return opt action=====#
        if (x, y) == (gx, gy):
          while parent[s] is not None and parent[parent[s]] is not None:
              s = parent[s]
          return int(act_from_parent[s])

        #=====Expand 3 successors from (x,y,d): turn left, turn right, move forward=====#
        #LEFT
        sL = (x,y,(d - 1)%4) #new state
        if g[s] + 1 < g.get(sL, 1e9): #does the path “current best cost to s” + cost of this action (1) beat the best cost we’ve seen for sL, if unseen 1e9 is large
            #if better...
            g[sL] = g[s] + 1 #update the best known cost-to-reach sL
            parent[sL] = s #backpointer
            act_from_parent[sL] = Actions.left #backpointer
            f = g[sL] + _heuristic(x, y, gx, gy, sL[2]) #compute the A* priority
            heappush(pq, (f, g[sL], random.random(), sL))
        #RIGHT
        sR = (x,y,(d + 1)%4) #new state
        if g[s] + 1 < g.get(sR, 1e9):
            g[sR] = g[s] + 1
            parent[sR] = s
            act_from_parent[sR] = Actions.right
            f = g[sR] + _heuristic(x, y, gx, gy, sR[2])
            heappush(pq, (f, g[sR], random.random(), sR))
        #FORWARD
        DX = (1, 0, -1, 0)  #0=right,1=down,2=left,3=up
        DY = (0, 1, 0, -1)
        nx, ny = x + DX[d], y + DY[d] #new state x, y, d remains same
        if _is_passable_cell(env, nx, ny):
            sF = (nx, ny, d)
            if g[s] + 1 < g.get(sF, 1e9):
                g[sF] = g[s] + 1
                parent[sF] = s
                act_from_parent[sF] = Actions.forward
                f = g[sF] + _heuristic(nx, ny, gx, gy, d) #heuristic now evaluated at the new cell (nx, ny) with same dir d
                heappush(pq, (f, g[sF], random.random(), sF))
    #if the heap empties with no goal found, stop safely
    return int(Actions.done)

from stable_baselines3.common.policies import BasePolicy
from imitation.policies.serialize import policy_registry

class AStarPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, env=None):
        super().__init__(observation_space, action_space)
        self.env = env  #need env for agent/goal positions

    def forward(self, obs, deterministic=False):
        return None

    def _predict(self, observation, deterministic=False):
        batch_size = observation.shape[0] if isinstance(observation, np.ndarray) else 1
        acts = []
        for _ in range(batch_size):
            base_env = self.env.envs[0].unwrapped
            agent_pos = tuple(base_env.agent_pos)
            goal_pos = getattr(base_env, "goal_pos", (9, 9))
            act = find_optimal_action(base_env, goal_pos=goal_pos, agent_pos=agent_pos)
            acts.append(act)
        return torch.as_tensor(acts, device=self.device)




