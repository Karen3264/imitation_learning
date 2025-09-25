from gymnasium.core import ObservationWrapper
from gymnasium import spaces
import numpy as np

class OneHotObsWrapper(ObservationWrapper):
    def __init__(self, env, tile_size=8):
        super().__init__(env)
        obs_shape = env.observation_space["image"].shape
        num_bits=8 # 4 types, 4 colors, 4 orientations for agent

        new_image_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")
        #type=0:agent originally 10
        #type=1:empty
        #type=2:wall
        #type=3:goal originally 8
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                state = img[i, j, 2]
                type_ = img[i, j, 0]

                if type_==10:
                    type_=0
                if type_==8:
                    type_=3

                out[i, j, type_] = 1
                out[i, j, 4 + state] = 1
        return {**obs, "image": out}
