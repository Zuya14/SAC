import gym
import pybullet_envs
from SAC import SAC
from trainer import Trainer

ENV_ID = 'InvertedPendulumBulletEnv-v0'
SEED = 0
REWARD_SCALE = 1.0
NUM_STEPS = 5 * 10 ** 4
EVAL_INTERVAL = 10 ** 3

env = gym.make(ENV_ID)
env_test = gym.make(ENV_ID)

algo = SAC(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    seed=SEED,
    reward_scale=REWARD_SCALE,
)

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
)

trainer.train()

trainer.plot()

trainer.visualize()
