import gym
import numpy as np
from collections import deque

class DroneControlEnv(gym.Env):
    def __init__(self):
        self.image = None
        self.queue_len = 4
        self.image_queue = None
        self._setup()
        self.reset()

    # This will setup the environment, put here any initialization of communication between AI and drone
    def setup(self):
        return

    # Returns a numpy array of the rgb pixel values (256, 144,3)
    def _get_rbg(self):
        # TODO Implement to get data from drone
        return rgb

    # Returns a numpy array of depth values (256, 144, 1)
    def _get_depth(self):
        # TODO Implement to get data from drone
        return depth

    # NOTE: We may not need to use this now but good to have for future use
    # Returns: a tuple of numpy arrays corresponding to:
    # o:    the orientation of the drone in degrees [roll, pitch, yaw]
    # v:    the velocity of the drone in m/s [x,y,z]
    # r:    the speed of rotation in degrees/s [roll, pitch, yaw]
    def _get_aux_info(self):
        return (o, v, r)

    def _get_obs(self):
        rgb = self._get_rbg()
        depth = self._get_depth()
        (o, v, r) = self._get_aux_info()
        self.image = np.concatenate([rgb, depth], axis=2)
        if self.image_queue is None:
            #self.last_image = self.image
            self.image_queue = deque([self.image]*self.queue_len)
        else:
            self.image_queue.append(self.image)
            self.image_queue.popleft()
        self.observation = np.concatenate(list(self.image_queue))
        self.observation = (self.observation.flatten())/255.0
        return self.observation

    # Figure out a way to send a kill signal to the AI so that it knows when it is finished
    def is_done(self):
        return done


    # The following functions are for use by the RL algorithm
    def step(self, raw_action):
        # Some processing of the [0-9] action sent, 0-8 corresponds to a position on a grid:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # 9 sends a None, implying that the drone is not in the frame
        if raw_action == 9:
            action = None
        else:
            action = [int(raw_action % 3), int(raw_action / 3)]
        done = self.is_done()
        obs = self._get_obs()
        rew = 0
        info = {}
        return obs, rew, done, info
    def reset(self):
        # TODO: Anything that may need to be done between multiple starts and stops of the AI
        return self._get_obs()