from typing import Callable, Optional, List, Union, Sequence, Type, Any

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvStepReturn,
    VecEnvObs,
)

from haxballgym.gym import Gym as GymEnv


def recording_condition(gym_env: GymEnv) -> bool:
    game = gym_env._match._game
    if game.score.red + game.score.blue > 0:
        return True

    return False


class SB3SingleInstanceEnv(VecEnv):
    """
    Class for wrapping a single haxballgym env into a VecEnv.
    """

    def __init__(
        self,
        env: GymEnv,
        recording_condition: Callable[[GymEnv], bool] = recording_condition,
    ):
        """
        :param env: the environment to wrap.
        """
        super().__init__(env._match.agents, env.observation_space, env.action_space)
        self.env = env
        self.step_result = None
        self.recording_condition = recording_condition

    def reset(self) -> VecEnvObs:
        save_recording = self.recording_condition(self.env)
        observations = self.env.reset(save_recording=save_recording)
        return np.asarray(observations)

    def step_async(self, actions: np.ndarray) -> None:
        if len(np.shape(actions)) == 1:
            actions = [actions]

        if self.env._match._bots is not None:
            match_env = self.env._match
            n_agents = match_env.agents
            game_env = match_env._game
            actions_bots = [p.step(game_env) for p in game_env.players][n_agents:]
            parsed_action_bots = []
            for act_bot in actions_bots:
                parsed_act = [act_bot[0] + 1, act_bot[1] + 1, act_bot[2]]
                parsed_action_bots.append(parsed_act)

            # The first action will always be the non bots
            actions_final = np.concatenate((actions, parsed_action_bots))
            self.step_result = self.env.step(actions_final)
        else:
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

            save_recording = self.recording_condition(self.env)
            observations = self.env.reset(save_recording=save_recording)

        else:
            infos = [info] * len(rewards)
        return (
            np.asarray(observations),
            np.array(rewards),
            np.full(len(rewards), done),
            infos,
        )

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return [self.env.seed(seed)] * self.num_envs

    # Bunch of functions that need to be overridden to work, might have to implement
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        pass

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        pass

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return [False] * self.num_envs

    def get_images(self) -> Sequence[np.ndarray]:
        pass
