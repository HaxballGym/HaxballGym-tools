from haxballgym import make, Game
from haxballgym import common_values as cv

from haxballgym.utils.terminal_conditions import common_conditions
from haxballgym.utils.reward_functions import (
    common_rewards,
    CombinedReward,
    velocity_reward,
)
from haxballgym.utils.obs_builders import DefaultObs

from haxballgym.utils.action_parsers import DefaultAction

import logging
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from haxballgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

if __name__ == "__main__":
    frame_skip = 10
    half_life_seconds = 5

    def create_gym_env():
        game = Game(
            stadium_file=cv.MAP_CLASSIC,
            folder_rec="recordings/",
            logging_level=logging.DEBUG,
        )
        gym_env = make(
            game=game,
            reward_fn=CombinedReward(
                (
                    common_rewards.EventReward(
                        team_goal=1, team_concede=-1, touch=0.01, kick=0.1
                    ),
                    velocity_reward.VelocityPlayerToBallReward(),
                    velocity_reward.VelocityBallToGoalReward(
                        stadium=game.stadium_store, own_goal=False
                    ),
                )
            ),
            terminal_conditions=(
                common_conditions.TimeoutCondition(1 * 60 * 60 / frame_skip),
                common_conditions.GoalScoredCondition(),
            ),
            obs_builder=DefaultObs(),
            action_parser=DefaultAction(),
            team_size=1,
            tick_skip=frame_skip,
        )
        return gym_env._match

    gym_env = [create_gym_env() for _ in range(8)]

    fps = 60 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))

    # wrap the RLGym environment with the single instance wrapper
    env = SB3MultipleInstanceEnv(gym_env)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    # Save a checkpoint every 50000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 / env.num_envs,
        save_path="logs/",
        name_prefix="hax_model_multi",
    )

    # create a PPO instance and start learning
    model = PPO(
        "MlpPolicy",
        env,
        n_epochs=32,  # PPO calls for multiple epochs
        learning_rate=1e-5,  # Around this is fairly common for PPO
        ent_coef=0.01,  # From PPO Atari
        vf_coef=1.0,  # From PPO Atari
        gamma=gamma,  # Gamma as calculated using half-life
        verbose=3,  # Print out all the info as we're going
        batch_size=4096,  # Batch size as high as possible within reason
        n_steps=4096,  # Number of steps to perform before optimizing network
        tensorboard_log="policy/",  # `tensorboard --logdir policy` in terminal
        device="auto",  # Uses GPU if available
    )

    model.learn(100_000_000, callback=checkpoint_callback)

    # model = PPO.load(
    #     "logs/hax_model_single_38000000_steps.zip",
    #     env,
    #     custom_objects=dict(
    #         n_envs=env.num_envs, _last_obs=None
    #     ),  # Need this to change number of agents
    #     device="auto",  # Need to set device again (if using a specific one)
    # )
    # env.reset()  # Important when loading models, SB3 does not do this for you

    # # Use reset_num_timesteps=False to keep going with same logger/checkpoints
    # model.learn(100_000_000, callback=checkpoint_callback, reset_num_timesteps=False)