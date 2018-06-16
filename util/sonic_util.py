"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Box


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class DeltaXReward(gym.Wrapper):
    def __init__(self, env):
        super(DeltaXReward, self).__init__(env)
        self.last_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self.last_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        rew = info['x']- self.last_x
        self.last_x = info['x']
        return obs, rew, done, info

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class BlackAndWhite(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 1-d
        tuple = modTupByIndex(self.observation_space.shape,2,1)
        self.observation_space = Box(0, 255,tuple)

    def _observation(self, observation):
        obs = observation.copy()
        # remove the score field, replace with white box
        obs[:51,:130,:] = np.ones((51,130,3)) * 255
        obs = self.rgb2gray(obs)
        obs = self.simple_threshold(obs,110)

#        plt.imshow(obs, aspect="auto",cmap='gray')
#        plt.show()

        return obs

    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def simple_threshold(self, im, threshold=128):
        return ((im > threshold) * 255).astype("uint8")

def sonicize_env(env):
    env = BlackAndWhite(env)
    env = AllowBacktracking(env)
    env = DeltaXReward(env)
    # wrap this such that only meaningful actions in Sonic can be performed
    env = SonicDiscretizer(env)
    return env

def modTupByIndex(tup, index, ins):
    lst = list(tup)
    lst[index] = ins
    return tuple(lst)