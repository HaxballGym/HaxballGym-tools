from typing import Optional, List, Union, Sequence, Type, Any

import gym
import numpy as np
from haxballgym.envs import Environment as HaxEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn, VecEnvObs

from haxballgym.gym import Gym

def recording_condition(env: HaxEnv) -> bool:
    # Returns whether the game should be recorded or not
    # Feel free to change the condition if you find that too many games are being recorded
    if env._match._game.score.red + env._match._game.score.blue > 0:
        return True
        
    return False

class SB3SingleInstanceEnv(VecEnv):
    """
    Class for wrapping a single haxballgym env into a VecEnv (each player is treated as its own environment).
    """

    def __init__(self, env: Gym):
        """
        :param env: the environment to wrap.
        """
        super().__init__(env._match.agents, env.observation_space, env.action_space)
        self.env = env
        self.step_result = None

    def reset(self) -> VecEnvObs:
        save_recording = recording_condition(self.env)
        observations = self.env.reset(save_recording=save_recording)
        return np.asarray(observations)

    def step_async(self, actions: np.ndarray) -> None:
        self.step_result = self.env.step(actions)

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, done, info = self.step_result

        if type(rewards) not in (tuple, list, np.ndarray):
            rewards = [rewards]
            observations = [observations]

        if done:
            # Following what SubprocVecEnv does
            infos = [info] * len(rewards)
            for info, obs in zip(infos, observations):
                info["terminal_observation"] = obs

            save_recording = recording_condition(self.env)
            observations = self.env.reset(save_recording=save_recording)

        else:
            infos = [info] * len(rewards)
        return np.asarray(observations), np.array(rewards), np.full(len(rewards), done), infos

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return [self.env.seed(seed)] * self.num_envs

    # Now a bunch of functions that need to be overridden to work, might have to implement later
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False] * self.num_envs

    def get_images(self) -> Sequence[np.ndarray]:
        pass

