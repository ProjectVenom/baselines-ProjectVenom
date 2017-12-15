from baselines.PythonClient import *
import time
import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame

from baselines.AirSimDisc import AirSimDisc
#from baselines.AirSimEnvFollow import AirSimEnv

def main():
    env = AirSimDisc()
    model = deepq.models.cnn_to_mlp_custom(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[4096, 4096, 4096],
        dueling=True
    )
    act = deepq.load("airsim_model.pkl", model, env)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
