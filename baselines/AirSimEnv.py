# -*- coding: utf-8 -*-
import math, io, random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from PIL import Image
from baselines.PythonClient import *
from baselines.projection import *
import tensorflow as tf

class AirSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.client = AirSimClient(port=41451)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.log_file = open('logs.txt', 'w')

        self.min_X = 0.0
        self.max_X = 1.0
        self.min_Y = 0.0
        self.max_Y = 1.0
        self.rt2 = math.sqrt(2)
        self.episodes = 0
        self.cumulative = 0.0

        MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        # PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
        #                                                            use_display_name=True)
        # category_index = label_map_util.create_category_index(categories)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)


        self._reset()
        # self.min_position = -1.2
        # self.max_position = 0.6
        # self.max_speed = 0.07
        # self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        # self.power = 0.0015

        # self.low_state = np.array([self.min_position, -self.max_speed])
        # self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None
        #elf.observation = self.image
        #self.observation = np.concatenate([self.last_image, self.image])
        # self.action_space = spaces.Box(0.0, 1.0, shape = (4,))
        self.action_space = spaces.Box(-0.5, 0.5, shape = (2,))
        #self.observation_space = spaces.Box(low=np.zeros(int(self.width),int(self.height),3), high=np.zeros(int(self.width),int(self.height),3)+255)
        self.observation_space = spaces.Box(low=np.zeros(self.observation.shape), high=np.zeros(self.observation.shape)+255)
        self.observation = None

        self._seed()

    def get_rbg(self, response):
        binary_rgb = response.image_data_uint8
        png = Image.open(io.BytesIO(binary_rgb)).convert('RGB')
        rgb = np.array(png)
        self.width = rgb.shape[1]
        self.height = rgb.shape[0]
        #rgb_vec = rgb.flatten()
        return rgb

    def get_depth(self, response):
        binary_rgb = response.image_data_uint8
        png = Image.open(io.BytesIO(binary_rgb)).convert('RGB')
        rgb = np.array(png)
        depth = np.expand_dims(rgb[:,:,0], axis=2)
        #w = Image.fromarray(depth, mode='L')
        #w.show()
        self.width = rgb.shape[1]
        self.height = rgb.shape[0]
        #depth_vec = depth.flatten()
        return depth

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obs(self, action=None):
        if self.image is None:
            return None
        #self.observation = np.concatenate([self.last_image, self.image])
        self.observation = self.image

        # 5 is airplane
        # 16 is bird
        rgb = np.array([np.transpose(np.transpose(self.image)[:3])])
        (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: rgb})

        plane_box = boxes[0][5][:]
        bird_box = boxes[0][16][:]
        maybe_plane_box = boxes[0][15][:]
        maybe_bird_box = boxes[0][16][:]
        self.observation = self.observation.flatten()
        if action is not None:
            a = np.array(action).flatten()
            self.observation = np.concatenate([a,
                                               plane_box, bird_box, maybe_plane_box, maybe_bird_box,
                                               self.observation])
        return self.observation

    def _step(self, raw_action):
        #action = np.matrix([raw_action.item(0)*float(self.width),raw_action.item(1)*float(self.height)])
        action = np.matrix([raw_action.item(0)+0.5,raw_action.item(1)+0.5])
        #x = self.c.item(0)/self.width
        #y = self.c.item(1)/self.height
        self.reward = 1-((np.linalg.norm(action-self.last_loc))/self.rt2)
        self.cumulative += self.reward
        self.iteration += 1
        #print(self.iteration)

        if self.episodes % 500 == 0:
            if self.fw is None:
                self.fw = open('./images/episode_'+str(self.episodes)+'/actions.txt', 'w')
            self.fw.write('('+str(action.item(0))+','+str(action.item(1))+')\n')

        # An action of 0 is the NOOP
        j = 0
        while True:
            if j > 50:
                self.done = True

                if self.episodes % 500 == 0:
                    self.fw.close()
                    self.fw = None
                self.episodes+=1
                print(str(self.episodes)+': '+str(self.cumulative/float(self.iteration))+' *'+str(self.iteration))
                self.log_file.write(str(self.episodes)+': '+str(self.cumulative/float(self.iteration))+' *'+str(self.iteration)+'\n')
                self.cumulative = 0
                return self.observation, self.reward, self.done, self.info
            rot_inc = 5.0+float(j)/10.0
            vel_inc = 1.0+float(j)/10.0
            #print(rot_inc)
            dC = np.matrix([random.normalvariate(self.v.item(0),vel_inc/self.fps),
                            random.normalvariate(self.v.item(1),vel_inc/self.fps),
                            random.normalvariate(self.v.item(2),vel_inc/self.fps)]
                           )
            dO = np.matrix([random.normalvariate(self.r.item(0),vel_inc/self.fps),
                            random.normalvariate(self.r.item(1),rot_inc/self.fps),
                            random.normalvariate(self.r.item(2),rot_inc/self.fps)]
                           )
            newC = np.add(self.c, dC)
            newO = np.add(self.o, dO)
            d = np.linalg.norm(self.t-newC)
            (x, y) = projection(self.t, newC, newO, w=float(self.width), h=float(self.height))
            total_v = np.linalg.norm(dC)
            if x <= float(self.width)*0.95 and x >= float(self.width)*0.05 and y <= float(self.height)*0.95 and y >= float(self.height)*0.05 \
                    and d > 3 and d < 30 and newC.item(2) < -2 \
                    and total_v*self.fps <= 30:
                break
            j += 1
        self.c = newC
        self.v = dC
        self.o = newO
        self.r = dO
        x = x/float(self.width)
        y = y/float(self.height)
        self.last_loc = np.matrix([x, y])
        self.state = self._render()
        self.observation = self.get_obs(self.last_loc)
        self.done = (self.iteration > 100)
        info = (self.c, self.v, self.o, self.r)
        self.info = {}
        #print(action)
        #print(np.matrix([x,y]))
        #print(self.reward)
        if self.done:
            if self.episodes % 500 == 0:
                self.fw.close()
                self.fw = None
            self.episodes+=1
            print(str(self.episodes)+': '+str(self.cumulative/float(self.iteration)))
            self.log_file.write(str(self.episodes)+': '+str(self.cumulative/float(self.iteration))+'\n')
            self.cumulative = 0
        return self.observation, self.reward, self.done, self.info

    def _reset(self):
        self.iteration = 0
        self.t = np.matrix([-10.0, 10.0, -10.0])
        self.o = np.matrix([0.0,0.0,0.0])
        self.c = np.matrix([-20.0, 10.0, -10.0])
        self.v = np.matrix([0.0,0.0,0.0])
        self.r = np.matrix([0.0,0.0,0.0])
        self.fps = 30.0
        self.client.simSetPose(Vector3r(self.c.item(0), self.c.item(1), self.c.item(2)), 
                               self.client.toQuaternion(math.radians(self.o.item(1)),math.radians(self.o.item(0)),math.radians(self.o.item(2))))
        self.image = None
        self.fw = None
        #response = self.client.simGetImages([ImageRequest(0, AirSimImageType.Scene)])[0]
        #self.image = self.get_rbg(response)

        self._render()

        self.observation = self.get_obs(np.matrix([0.5,0.5]))

        (x, y) = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        x = x/float(self.width)
        y = y/float(self.height)
        self.last_loc = np.matrix([x, y])
        return self.observation

    def _render(self, mode='human', close=False):
        self.client.simSetPose(Vector3r(self.c.item(0), self.c.item(1), self.c.item(2)),
                               self.client.toQuaternion(math.radians(self.o.item(1)),math.radians(self.o.item(0)),math.radians(self.o.item(2))))

        self.last_image = self.image
        responses = self.client.simGetImages([ImageRequest(0, AirSimImageType.Scene),
                                              ImageRequest(0, AirSimImageType.DepthVis)])
        if self.episodes % 500 == 0:
            if not os.path.exists('./images/episode_'+str(self.episodes)+'/'):
                os.makedirs('./images/episode_'+str(self.episodes)+'/')
            AirSimClient.write_file(os.path.normpath('./images/episode_'+str(self.episodes)+'/'+str(self.iteration)+'.png'),
                                    responses[0].image_data_uint8)
        rgb = self.get_rbg(responses[0])
        #response = self.client.simGetImages([ImageRequest(0, AirSimImageType.DepthVis)])[0]
        depth = self.get_depth(responses[1])
        self.image = np.concatenate([rgb, depth], axis=2)

        if self.last_image is None:
            self.last_image = self.image

        return self.image

