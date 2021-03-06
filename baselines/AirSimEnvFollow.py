# -*- coding: utf-8 -*-
import math, io, random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from PIL import Image
from baselines.PythonClient import *
from baselines.projection import *
from collections import deque


class AirSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, discrete=True):
        this_port = 41450
        print(this_port)

        self.client = AirSimClient(port=this_port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.target = AirSimClient(port=41451)
        self.target.confirmConnection()
        self.target.enableApiControl(True)
        self.target.armDisarm(True)

        self.log_file = open('logs.txt', 'w')
        self.acc_file = open('accs.txt', 'w')

        self.episodes = 0
        self.fps = 60
        self.max_iter = 15*self.fps
        self.queue_len = 4

        self.t = np.matrix([-10.0, 10.0, -10.0])
        self.o = np.matrix([0.0, 0.0, 0.0])
        self.c = np.matrix([-20.0, 10.0, -10.0])
        self.v = np.matrix([0.0, 0.0, 0.0])
        self.r = np.matrix([0.0, 0.0, 0.0])
        self.image = None
        self.image_queue = None
        self.iteration = 0

        self.target.simSetPose(Vector3r(self.t.item(0), self.t.item(1), self.t.item(2)),
                               self.client.toQuaternion(0,0,0))
        self._render()
        self.reset()

        self.viewer = None
        self.discrete = discrete
        if self.discrete:
            self.action_space = spaces.Discrete(81)
            self.observation_space = spaces.Box(low=np.zeros(self.current_state.shape) - 1,
                                                high=np.zeros(self.current_state.shape) + 1)
        else:
            self.action_space = spaces.Box(-1, 1, shape = (4,))
            self.observation_space = spaces.Box(low=np.zeros(self.current_state.shape) - 1,
                                            high=np.zeros(self.current_state.shape) + 1)
        self.current_state = None
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0
        self._seed()

    ##
    ## Helper Functions ##
    ##
    def _random_orientation(self, t):
        i = 51
        while i > 50:
            while True:
                c = np.matrix([random.normalvariate(t.item(0), 10),
                               random.normalvariate(t.item(1), 10),
                               random.normalvariate(t.item(2), 5)])
                d = np.linalg.norm(t - c)
                if d > 10 and d < 15 and c.item(2) < -5:
                    #    break
                    # while True:
                    # o = np.matrix([random.uniform(-180,180),
                    #               random.uniform(-180,180),
                    #               random.uniform(-180,180)])
                    o = get_o_from_pts(t, c)
                    (x, y), target_in_front = projection(t, c, o, w=float(self.width), h=float(self.height))
                    if x == self.width / 2 and y == self.height / 2 and target_in_front:
                        break

            r = np.matrix([0.0, 0.0, 0.0])
            v = np.matrix([0.0, 0.0, 0.0])
            #print((x, y))

            for i in range(50):
                j = 0
                while True:
                    rot_inc = 5.0 + float(j) / 10.0
                    vel_inc = 10.0 + float(j) / 10.0
                    if j > 50:
                        break
                    # print(rot_inc)
                    dC = np.matrix([random.normalvariate(v.item(0), vel_inc / self.fps),
                                    random.normalvariate(v.item(1), vel_inc / self.fps),
                                    random.normalvariate(v.item(2), vel_inc / self.fps)]
                                   )
                    dO = np.matrix([random.normalvariate(r.item(0), vel_inc / self.fps),
                                    random.normalvariate(r.item(1), rot_inc / self.fps),
                                    random.normalvariate(r.item(2), rot_inc / self.fps)]
                                   )
                    newC = np.add(c, dC)
                    newO = np.add(o, dO)
                    d = np.linalg.norm(self.t - newC)
                    (x, y), target_in_front = projection(self.t, newC, newO, w=float(self.width),
                                                         h=float(self.height))
                    total_v = np.linalg.norm(dC)
                    if x <= float(self.width) * 0.95 and x >= float(self.width) * 0.05 and y <= float(
                            self.height) * 0.95 and y >= float(self.height) * 0.05 \
                            and d > 10 and d < 15 and newC.item(2) < -5 \
                            and total_v * self.fps <= 30 \
                            and target_in_front:
                        break
                    j += 1
                c = newC
                v = dC
                o = newO
                r = dO
            if x <= float(self.width) * 0.95 and x >= float(self.width) * 0.05 and y <= float(
                    self.height) * 0.95 and y >= float(self.height) * 0.05 \
                    and d > 10 and d < 15 and c.item(2) < -5 \
                    and target_in_front:
                    break
        self.last_d = np.linalg.norm(self.t - c)

        (x, y), target_in_front = projection(self.t, c, o, w=float(self.width), h=float(self.height))
        #print((x, y))
        return (c, o)

    def _get_rgb(self, response):
        binary_rgb = response.image_data_uint8
        png = Image.open(io.BytesIO(binary_rgb)).convert('RGB')
        rgb = np.array(png)
        self.width = rgb.shape[1]
        self.height = rgb.shape[0]
        # rgb_vec = rgb.flatten()
        return rgb

    def _get_depth(self, response):
        binary_rgb = response.image_data_uint8
        png = Image.open(io.BytesIO(binary_rgb)).convert('RGB')
        rgb = np.array(png)
        depth = np.expand_dims(rgb[:, :, 0], axis=2)
        # w = Image.fromarray(depth, mode='L')
        # w.show()
        self.width = rgb.shape[1]
        self.height = rgb.shape[0]
        # depth_vec = depth.flatten()
        return depth

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        if self.image is None:
            return None

        #self.current_state = np.concatenate(list(self.image_queue))
        #self.current_state = (self.current_state.flatten())
        (x, y), target_in_front = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        pix = np.array((x/255.0,y/143.0)).flatten()
        self.current_state = np.concatenate([np.array(self.v).flatten()/(10.0/self.fps),
                                             np.array(self.o).flatten()/360.0,
                                             np.array(self.r).flatten()/360.0,
                                             np.array(self.t).flatten()/30.0,
                                             np.array(self.c).flatten()/30.0,
                                             np.array(pix),
                                             self.current_state], 0)

        return self.current_state

    def _render(self, mode='human', close=False):
        self.client.simSetPose(Vector3r(self.c.item(0), self.c.item(1), self.c.item(2)),
                               self.client.toQuaternion(math.radians(self.o.item(1)), math.radians(self.o.item(0)),
                                                        math.radians(self.o.item(2))))

        self.last_image = self.image
        responses = self.client.simGetImages([ImageRequest(0, AirSimImageType.Scene),
                                              ImageRequest(0, AirSimImageType.DepthVis)])
        if self.episodes % 100 == 0:
            if not os.path.exists('./images/episode_' + str(self.episodes) + '/'):
                os.makedirs('./images/episode_' + str(self.episodes) + '/')
            AirSimClient.write_file(
                os.path.normpath('./images/episode_' + str(self.episodes) + '/' + str(self.iteration) + '.png'),
                responses[0].image_data_uint8)
        rgb = self._get_rgb(responses[0])
        # response = self.client.simGetImages([ImageRequest(0, AirSimImageType.DepthVis)])[0]
        depth = self._get_depth(responses[1])
        self.image = np.concatenate([rgb, depth], axis=2)
        # self.image = rgb
        #if self.last_image is None:
        #    self.last_image = self.image

        if self.image_queue is None:
            #self.last_image = self.image
            self.image_queue = deque([self.image]*self.queue_len)
        else:
            self.image_queue.append(self.image)
            self.image_queue.popleft()


        return self.image

    def get_num_actions(self):
        return self.game.env.action_space.n

    def step(self, raw_action):
        # action = np.matrix([raw_action.item(0)*float(self.width),raw_action.item(1)*float(self.height)])
        # x = self.c.item(0)/self.width
        # y = self.c.item(1)/self.height
        # self.reward = 1-((np.linalg.norm(action-self.last_loc))/self.rt2)
        if self.discrete:
            # action = [int(raw_action % 3), int(raw_action / 3)]
            action = raw_action
            yaw = 1 - action % 3
            action = int(action / 3)
            pitch = 1 - action % 3
            action = int(action / 3)
            roll = 1 - action % 3
            action = int(action / 3)
            acc = 1 - action % 3
        else:
            action = raw_action
            roll = action.item(0)
            pitch = action.item(1)
            yaw = action.item(2)
            acc = action.item(3)

        # print(self.iteration)

        if self.episodes % 100 == 0:
            if self.fw is None:
                self.fw = open('./images/episode_' + str(self.episodes) + '/actions.txt', 'w')
            self.fw.write('(' + str(action) + ')\n')

        # An action of 0 is the NOOP
        j = 0

        max_v = 10.0
        max_r = 360.0
        dR = np.matrix([roll, pitch, yaw])
        self.v = self.v + acc / self.fps
        if self.v*self.fps > max_v: self.v = max_v/self.fps
        self.r = self.r + dR / self.fps

        if self.r.item(0) > max_r: self.r = np.matrix([self.r.item(0)-max_r, self.r.item(1), self.r.item(2)])
        if self.r.item(1) > max_r: self.r = np.matrix([self.r.item(0), self.r.item(1)-max_r, self.r.item(2)])
        if self.r.item(2) > max_r: self.r = np.matrix([self.r.item(0), self.r.item(1), self.r.item(2)-max_r])

        direction = np.dot(rot_mat(self.r.item(0), self.r.item(1), self.r.item(2)), np.transpose(self.c))
        direction = np.transpose(direction) / np.linalg.norm(direction)
        self.o = self.o + self.r
        if self.o.item(0) > max_r: self.o = np.matrix([self.o.item(0)-max_r, self.o.item(1), self.o.item(2)])
        if self.o.item(1) > max_r: self.o = np.matrix([self.o.item(0), self.o.item(1)-max_r, self.o.item(2)])
        if self.o.item(2) > max_r: self.o = np.matrix([self.o.item(0), self.o.item(1), self.o.item(2)-max_r])
        self.c = self.c + direction * self.v
        if self.c.item(2) > -5:
            self.c = np.matrix([self.c.item(0), self.c.item(1), -5])

        self.state = self._render()

        self.previous_state = self.current_state
        self.current_state = self._get_obs()

        (x, y), target_in_front = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        pix_d = ((x - self.width / 2) ** 2 + (y - self.height / 2) ** 2) / (
        (self.width / 2) ** 2 + (self.height / 2) ** 2)

        d = np.linalg.norm(self.t - self.c)
        d_max = 0

        # d_norm = d/d_max

        center_reward = 1 - pix_d
        dist_reward = 1 / d

        # self.reward = dist_reward
        # if d < self.last_d:
        #    self.reward = 1
        # elif d > self.last_d:
        #    self.reward = -1
        # else:
        #    self.reward = 0
        #self.reward = 0
        #self.reward = (30.0/d - 30.0/self.last_d)
        #print(self.reward)
        self.reward = (self.last_d - d)
        self.last_d = d
        #self.last_d = min(d, self.last_d)
        self.done = (self.iteration > self.max_iter)

        if x > self.width or x < 0 or y > self.height or y < 0 or d > 30:
            print((x,y))
            self.done = True
            self.reward = 0

        if d < 1:
            self.done = True
            self.reward = 0

        # self.reward -= 0.1

        self.total_reward += self.reward
        self.iteration += 1

        if self.done:
            # print(self.c)
            if self.episodes % 100 == 0:
                self.fw.close()
                self.fw = None
            self.episodes += 1
            print(str(self.episodes) + ': ' + str(self.total_reward)+ ' Ended At: ' + str(self.reward) + ' @ ' + str(float(self.iteration)) + ' with d: '+str(d) )
            self.log_file.write(str(self.episodes) + ': ' + str(self.total_reward)+ ' Ended At: ' + str(self.reward) + ' @ ' + str(float(self.iteration))  + ' with d: '+str(d) + '\n')

            self.total_reward = 0
        return self.current_state, self.reward, self.done, {}

    def reset(self):
        self.iteration = 0
        self.t = np.matrix([-10.0, 10.0, -10.0])
        self.o = np.matrix([0.0, 0.0, 0.0])
        self.c = np.matrix([-20.0, 10.0, -10.0])
        self.v = 0.0
        self.r = np.matrix([0.0, 0.0, 0.0])
        self.nb_correct = 0
        self.image = None
        self.fw = None
        self.c, self.o = self._random_orientation(t)
        self.last_d = np.linalg.norm(self.t - self.c)
        self.target.simSetPose(Vector3r(self.t.item(0), self.t.item(1), self.t.item(2)),
                               self.client.toQuaternion(0,0,0))
        self._render()

        self.current_state = self._get_obs()
        return self.current_state
