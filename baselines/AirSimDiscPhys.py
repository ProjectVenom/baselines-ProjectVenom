# -*- coding: utf-8 -*-
import math, io, random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from PIL import Image
from baselines.PythonClient import *
from baselines.projection import *


class AirSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, hunter_port=41451, target_port = 41450):
        self.hunter = AirSimClient(port=hunter_port)
        self.hunter.confirmConnection()
        self.hunter.enableApiControl(True)
        self.hunter.armDisarm(True)
        #self.hunter.takeoff()
        self.hunter.simSetPose(Vector3r(-20, 10, -10), self.hunter.toQuaternion(0, 0, 0))

        self.target = AirSimClient(port=target_port)
        self.target.confirmConnection()
        self.target.enableApiControl(True)
        self.target.armDisarm(True)
        #self.target.takeoff()
        self.target.simSetPose(Vector3r(-20, 10, -10), self.target.toQuaternion(0, 0, 0))

        self.log_file = open('logs.txt', 'w')
        self.acc_file = open('accs.txt', 'w')

        self.min_X = 0.0
        self.max_X = 1.0
        self.min_Y = 0.0
        self.max_Y = 1.0
        self.rt2 = math.sqrt(2)
        self.episodes = 0
        self.cumulative = 0.0
        self.max_iter = 100

        self.t = np.matrix([-20.0, -5.0, -10.0])
        self.o = np.matrix([0.0, 0.0, 0.0])
        self.c = np.matrix([-30.0, -5.0, -10.0])
        self.c = np.matrix([-30.0, -5.0, -10.0])
        self.vC = np.matrix([0.0, 0.0, 0.0])
        self.vT = np.matrix([0.0, 0.0, 0.0])
        self.image = None
        self.iteration = 0
        self.log_int = 100

        self._render()
        self._reset()
        # self.min_position = -1.2
        # self.max_position = 0.6
        # self.max_speed = 0.07
        # self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        # self.power = 0.0015

        # self.low_state = np.array([self.min_position, -self.max_speed])
        # self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None
        # elf.observation = self.image
        # self.observation = np.concatenate([self.last_image, self.image])
        # self.action_space = spaces.Box(0.0, 1.0, shape = (4,))
        # self.action_space = spaces.Box(-0.5, 0.5, shape = (2,))
        self.action_space = spaces.Discrete(9)

        # self.observation_space = spaces.Box(low=np.zeros(int(self.width),int(self.height),3), high=np.zeros(int(self.width),int(self.height),3)+255)
        self.observation_space = spaces.Box(low=np.zeros(self.observation.shape),
                                            high=np.zeros(self.observation.shape) + 255)
        self.observation = None

        self._seed()
    def random_orientation(self, t):
        while True:
            stop = False
            c = np.matrix([random.normalvariate(t.item(0), 10),
                           random.normalvariate(t.item(1), 10),
                           t.item(2)])
            d = np.linalg.norm(t - c)
            if d > 10 and d < 12 and c.item(2) < -2:
                #o = get_o_from_pts(t, c)
                dx = c.item(0) - t.item(0)
                dy = c.item(1) - t.item(1)
                yaw = math.degrees(math.asin(-dy/d))
                yaw2 = math.degrees(math.acos(-dx/d))
                if yaw != yaw2: stop = True
                o = np.matrix([0.0,0.0,yaw])
                (x, y), target_in_front = projection(t, c, o, w=float(self.width), h=float(self.height))
                if x <= float(self.width) * 0.95 and x >= float(self.width) * 0.05 and y <= float(
                        self.height) * 0.95 and y >= float(self.height) * 0.05 and target_in_front\
                        and not stop:
                    while True:
                        newC = np.matrix([random.normalvariate(c.item(0), d/2),
                                          c.item(1),
                                          random.normalvariate(c.item(2), d/2)])
                        (x, y), target_in_front = projection(t, newC, o, w=float(self.width), h=float(self.height))
                        if x <= float(self.width) * 0.95 and x >= float(self.width) * 0.05 and \
                            y <= float(self.height) * 0.95 and y >= float(self.height) * 0.05 and \
                                newC.item(2) < -2 and target_in_front and d > 10 and d < 15 and c.item(2) < -2:
                            self.c = newC
                            self.o = o
                            (x, y), target_in_front = projection(self.t, self.c, self.o,
                                                                 w=float(self.width),
                                                                 h=float(self.height))
                            #print((x, y))
                            return (self.c, self.o)

    def get_rbg(self, response):
        binary_rgb = response.image_data_uint8
        png = Image.open(io.BytesIO(binary_rgb)).convert('RGB')
        rgb = np.array(png)
        self.width = rgb.shape[1]
        self.height = rgb.shape[0]
        # rgb_vec = rgb.flatten()
        return rgb

    def get_depth(self, response):
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

    def get_obs(self, action=None):
        if self.image is None:
            return None

        self.observation = self.image
        #self.observation = np.concatenate([self.last_image, self.image])

        #self.observation = self.observation.flatten()
        #if action is not None:
        #   a = np.array(action).flatten()
        #   self.observation = np.concatenate([a, self.observation])

        return self.observation

    # Mapping
    # 0 1 2
    # 3 4 5
    # 6 7 8
    def _step(self, raw_action):
        # action = np.matrix([raw_action.item(0)*float(self.width),raw_action.item(1)*float(self.height)])
        # x = self.c.item(0)/self.width
        # y = self.c.item(1)/self.height
        # self.reward = 1-((np.linalg.norm(action-self.last_loc))/self.rt2)
        raw_action = raw_action - 1
        action = [int(raw_action % 3), int(raw_action / 3)]
        (x0, y0), target_in_front = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        self.c = self.hunter.getPosition()
        self.c = np.matrix([self.c.x_val, self.c.y_val, self.c.z_val])
        #self.vC = self.hunter.getVelocity()
        #self.vC = np.matrix([self.vC.x_val, self.vC.y_val, self.vC.z_val])
        self.o = self.hunter.getOrientation()
        self.o = np.matrix([math.degrees(self.o.x_val),
                            math.degrees(self.o.y_val),
                            2*math.degrees(self.o.z_val)])
        self.t = self.target.getPosition()
        self.t = np.matrix([self.t.x_val, self.t.y_val, self.t.z_val])
        #self.vT = self.target.getVelocity()
        #self.vT = np.matrix([self.vT.x_val, self.vT.y_val, self.vT.z_val])
        (x, y), target_in_front = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        d = np.linalg.norm(self.t - self.c)
        #print((x,y))
        #print((x0,y0))
        #print()
        self.iteration += 1
        if x < 0 or y < 0 or x > self.width or y > self.height or \
                self.t.item(2) > -2 or self.c.item(2) > -2 or \
                d < 5 or d > 25 or not target_in_front:
            self.done = True
        if not self.done:
            x = 3 * x / float(self.width)
            y = 3 * y / float(self.height)
            loc = [int(x),int(y)]
            #print(action)
            #print(loc)

            self.reward = 1 - ((action[0] - loc[0]) ** 2 + (action[1] - loc[1]) ** 2)
            if self.reward == 1:
                self.nb_correct += 1
            self.cumulative += self.reward
            # print(self.iteration)

            if self.episodes % self.log_int == 0:
                if self.fw is None:
                    self.fw = open('./images/episode_' + str(self.episodes) + '/actions.txt', 'w')
                self.fw.write('(' + str(raw_action) + ')\n')

            # An action of 0 is the NOOP

            self.aC = np.matrix([random.normalvariate(mu=self.aC.item(0), sigma=2/self.fps),
                            random.normalvariate(mu=self.aC.item(0), sigma=2/self.fps),
                           random.normalvariate(mu=self.aC.item(0), sigma=2/self.fps)]
                          )
            self.aT= np.matrix([random.normalvariate(mu=self.aT.item(0), sigma=2/self.fps),
                           random.normalvariate(mu=self.aT.item(0), sigma=2/self.fps),
                           random.normalvariate(mu=self.aT.item(0), sigma=2/self.fps)]
                          )
            self.vC = self.vC+self.aC
            self.vT = self.vT+self.aT
            newC = self.c + self.vC
            newT = self.t + self.vT
            #self.hunter.moveToPosition(newC.item(0), newC.item(1), newC.item(2), 1)
            #self.target.moveToPosition(newT.item(0), newT.item(1), newT.item(2), np.linalg.norm(self.vT))
            #self.hunter.moveByVelocity(self.vC.item(0),
            #                           self.vC.item(1),
            #                           self.vC.item(2),
            #                           10)
            #self.hunter.simSetPose(newC, self.hunter.getOrientation())
            #v_temp = self.hunter.getVelocity()
            #v_temp = np.matrix([v_temp.x_val, v_temp.y_val, v_temp.z_val])
            #v_magnitude = np.linalg.norm(v_temp)
            #if v_magnitude == 0:
            #    print(self.iteration)
            #self.target.moveByVelocity(self.vT.item(0),
            #                           self.vT.item(1),
            #                           self.vT.item(2),
            #                           10)
            #print(self.vT)
            self.t = newT
            self.target.simSetPose(Vector3r(newT.item(0), newT.item(1), newT.item(2)),
                               self.target.toQuaternion(0,0,0))
            #time.sleep(1)

            self.state = self._render()
            self.observation = self.get_obs()
            self.done = (self.iteration > self.max_iter)

        # Proper Termination
        if self.done:
            if self.episodes % self.log_int == 0 and self.fw is not None:
                self.fw.close()
                self.fw = None
            self.episodes += 1
            acc = float(self.nb_correct) / float(self.iteration)
            print(str(self.episodes) + ': ' + str(self.cumulative / float(self.iteration)) + ' len: '+str(self.iteration))
            self.log_file.write(str(self.episodes) + ': ' + str(self.cumulative / float(self.iteration)) + ' len: '+str(self.iteration) + '\n')
            print('Accuracy: ' + str(acc))
            self.acc_file.write(str(acc) + '\n')
            self.nb_correct = 0
            self.cumulative = 0
        return self.observation, self.reward, self.done, {}

    def _reset(self):
        #time.sleep(1)
        self.iteration = 0
        self.t = np.matrix([-20.0, -5.0, -10.0])
        self.o = np.matrix([0.0, 0.0, 0.0])
        self.c = np.matrix([-30.0, -5.0, -10.0])
        self.vC = np.matrix([0.0, 0.0, 0.0])
        self.vT = np.matrix([0.0, 0.0, 0.0])
        self.aC = np.matrix([0.0, 0.0, 0.0])
        self.aT = np.matrix([0.0, 0.0, 0.0])
        self.r = np.matrix([0.0, 0.0, 0.0])
        self.fps = 60.0
        self.nb_correct = 0
        self.done = False
        self.reward = 0.0
        #self.hunter.simSetPose(Vector3r(self.c.item(0), self.c.item(1), self.c.item(2)),
        #                       self.hunter.toQuaternion(math.radians(self.o.item(1)), math.radians(self.o.item(0)),
        #                                                math.radians(self.o.item(2))))
        self.image = None
        self.fw = None
        # response = self.hunter.simGetImages([ImageRequest(0, AirSimImageType.Scene)])[0]
        # self.image = self.get_rbg(response)

        t = np.matrix([random.normalvariate(self.t.item(0), 5),
                            random.normalvariate(self.t.item(1), 5),
                            random.normalvariate(self.t.item(0), 5)])
        while t.item(2) > -5:
            t = np.matrix([random.normalvariate(self.t.item(0), 5),
                                random.normalvariate(self.t.item(1), 5),
                                random.normalvariate(self.t.item(0), 5)])
        self.t = t
        self.c, self.o = self.random_orientation(self.t)
        self.hunter.simSetPose(Vector3r(self.c.item(0), self.c.item(1), self.c.item(2)),
                               self.hunter.toQuaternion(math.radians(self.o.item(1)), math.radians(self.o.item(0)),
                                                        math.radians(self.o.item(2))))
        self.hunter.moveByVelocity(0, 0, 1, 0.1)
        self.target.simSetPose(Vector3r(self.t.item(0), self.t.item(1), self.t.item(2)),
                               self.target.toQuaternion(0,0,0))
        self.target.moveByVelocity(0, 0, 1, 0.1)
        self._render()
        #self.hunter.simSetPose(Vector3r(self.c.item(0), self.c.item(1), self.c.item(2)),
        #                       self.hunter.toQuaternion(math.radians(self.o.item(1)), math.radians(self.o.item(0)),
        #                                                math.radians(self.o.item(2))))

        (x, y), _ = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        #print((x, y, _))
        x = 3 * x / float(self.width)
        y = 3 * y / float(self.height)
        self.observation = self.get_obs()
        return self.observation

    def _render(self, mode='human', close=False):
        self.last_image = self.image
        responses = self.hunter.simGetImages([ImageRequest(0, AirSimImageType.Scene),
                                              ImageRequest(0, AirSimImageType.DepthVis)])
        if self.episodes % self.log_int == 0:
            if not os.path.exists('./images/episode_' + str(self.episodes) + '/'):
                os.makedirs('./images/episode_' + str(self.episodes) + '/')
            AirSimClient.write_file(
                os.path.normpath('./images/episode_' + str(self.episodes) + '/' + str(self.iteration) + '.png'),
                responses[0].image_data_uint8)
        rgb = self.get_rbg(responses[0])
        # response = self.hunter.simGetImages([ImageRequest(0, AirSimImageType.DepthVis)])[0]
        depth = self.get_depth(responses[1])
        self.image = np.concatenate([rgb, depth], axis=2)
        # self.image = rgb
        if self.last_image is None:
            self.last_image = self.image

        return self.image
