import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

def run_episode_get_frame(env, seed: int = 0, policy=None, noise=False):
    trajectory = []
    obs, info = env.reset(seed=seed)
    trajectory.append(np.array(env.agent_pos))
    terminated = truncated = False
    while not (terminated or truncated):
        if noise == True:
          if random.random() < 0.1:
            action = env.action_space.sample()
          else: 
            action = policy.predict(obs)[0]

        else:
          action = env.action_space.sample()
          if policy is not None:
              action = policy.predict(obs)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(np.array(env.agent_pos))
    #frame = env.render()
    frame = env.unwrapped.render(highlight=False)
    env.close()

    return frame, np.array(trajectory)

def plot_k_trajectories(env, seeds, policy=None, corner_noise: float = 0.1, noise=False):
    all_trajs = []
    frame = None
    for seed in seeds:
        frame, traj = run_episode_get_frame(env, seed=seed, policy=policy, noise=noise)
        all_trajs.append(traj)

    cell_size = 32
    fig, ax = plt.subplots()
    ax.imshow(frame)

    cmap = cm.get_cmap("tab10", len(all_trajs))
    for i, traj in enumerate(all_trajs):
        xs = (traj[:, 0] + 0.5) * cell_size
        ys = (traj[:, 1] + 0.5) * cell_size
        ys = frame.shape[0] - ys 
        x, y = traj[0]
        xs = (x + 0.5) * cell_size
        ys = (y+ 0.5) * cell_size
        xs_prev=xs
        ys_prev=ys

        for j, agent_pos in enumerate(traj):
          x, y = agent_pos
          xs = (x+0.5) * cell_size 
          ys = (y+0.5) * cell_size 

          if j < len(traj) - 1 and j>0:
            x_next, y_next = traj[j+1]
            x_prev, y_prev=traj[j-1]
            if not (x==x_next and y==y_next) and not (x==x_prev and y==y_prev) :
              xs=xs+np.random.normal(scale=2)
              ys=ys+np.random.normal(scale=2)
          line = Line2D([xs_prev, xs ], [ys_prev, ys], color=cmap(i), linewidth=6, alpha=0.4)
          ax.add_line(line)
          xs_prev, ys_prev = xs, ys
    ax.axis("off")
    plt.show()


