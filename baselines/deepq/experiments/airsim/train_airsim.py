from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
import baselines.AirSimDiscPhys as AirSimDiscPhys
from baselines.AirSimDisc import AirSimDisc
from baselines.AirSimEnvFollow import AirSimEnv
from baselines.AirSimPhysFollow import AirSimPhys

def main():
    env = AirSimPhys()
    #env = AirSimDisc()
    #env = AirSimDiscPhys.AirSimEnv()
    #env = ScaledFloatFrame(wrap_dqn(env))
    model = deepq.models.cnn_to_mlp_custom(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[4096, 4096, 4096],
        dueling=True
    )
    # 2,000,000 original
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=1000000,
        buffer_size=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        checkpoint_freq=1000,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True
    )
    act.save("airsim_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
