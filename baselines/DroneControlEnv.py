import gym
import numpy as np
from collections import deque


class ImageReader:
    def __init__(self):
        self.__bridge = CvBridge()
        self.__rgb_sub = rospy.Subscriber("/zed/rgb/image_rect_color",Image,self.__rgb_cb)
        self.__depth_sub = rospy.Subscriber("/zed/depth/depth_registered",Image,self.__depth_cb)
        self.rgb_image = None
        self.depth_image = None
        rospy.init_node('image_reader', anonymous=True)

    def __rgb_cb(self,data):
        try:
            self.rgb_image = self.__bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def __depth_cb(self,data):
        try:
            self.depth_image = self.__bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
        except CvBridgeError as e:
            print(e)

    def get_rgb(self, width = 256, height = 144):
        rgb = cv2.resize(self.rgb_image,dsize =(width, height) )
        return rgb

    def get_depth(self, width = 256, height = 144):
        depth = cv2.resize(self.depth_image,dsize =(width, height) )
        return depth

class DroneControlEnv(gym.Env):
    def __init__(self):
        self.image = None
        self.queue_len = 4
        self.image_queue = None
        self._setup()
        self.reset()
    # This will setup the environment, put here any initialization of communication between AI and drone
    def setup(self): 
        self.ir = ImageReader()
        
    # Returns a numpy array of the rgb pixel values (256, 144,3)
    def _get_rbg(self):
        rgb = self.ir.get_rgb()
        return rgb
    # Returns a numpy array of depth values (256, 144, 1)
    def _get_depth(self):
        depth = self.ir.get_depth()
        return depth
    
    # NOTE: We may not need to use this now but good to have for future use
    # Returns: a tuple of numpy arrays corresponding to:
    # o:    the orientation of the drone in degrees [roll, pitch, yaw]
    # v:    the velocity of the drone in m/s [x,y,z]
    # r:    the speed of rotation in degrees/s [roll, pitch, yaw]
    def _get_aux_info(self):
        o = None
        v = None
        r = None
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
        action = [int(raw_action % 3), int(raw_action / 3)]
        done = self.is_done()
        obs = self._get_obs()
        rew = 0
        info = {}
        return obs, rew, done, info
    def reset(self):
        # TODO: Anything that may need to be done between multiple starts and stops of the AI
