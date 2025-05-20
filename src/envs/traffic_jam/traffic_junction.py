from envs.multiagentenv import MultiAgentEnv
from envs.traffic_jam.ic3net_envs.traffic_junction_env import TrafficJunctionEnv
import numpy as np
import gym
import argparse
import torch
from gym.spaces import flatdim
from gym.spaces.discrete import Discrete
from gym.wrappers import TimeLimit as GymTimeLimit


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
                self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all([done])
            done = len(observation) * [True]
        return observation, reward, done, info


class Traffic_JunctionEnv(MultiAgentEnv):

    def __init__(self,
                 nagents: int,
                 display: bool,
                 dim: int,
                 vision: int,
                 add_rate_min: float,
                 add_rate_max: float,
                 curr_start: float,
                 curr_end: float,
                 difficulty: str,
                 seed: int,
                 vocab_type: str,
                 map_name: str = ' ',
                 time_limit=50,
                 gas_delay=0,
                 brake_delay=0,
                 gas_std=0,
                 brake_std=0,
                 **kwargs,
                 ):

        parser_env = argparse.ArgumentParser('Example GCCNet environment random agent')
        parser_env.add_argument('--nagents', type=int, default=2, help="Number of agents")

        self.env = TrafficJunctionEnv()
        self.display = display
        if self.display:
            self.env.init_curses()
        self.env.init_args(parser_env)
        args_env = parser_env.parse_known_args()
        args_env = args_env[0]
        args_env.nagents = nagents
        args_env.dim = dim
        args_env.vision = vision
        # args_env.vision = brake_delay + 1
        args_env.add_rate_min = add_rate_min
        args_env.add_rate_max = add_rate_max
        args_env.curr_start = curr_start
        args_env.curr_end = curr_end
        args_env.difficulty = difficulty
        args_env.vocab_type = vocab_type

        self.env.multi_agent_init(args_env)
        self.episode_limit = time_limit
        self.env = TimeLimit(self.env, max_episode_steps=self.episode_limit)
        self.nagents = nagents

        self.gas_delay = gas_delay
        self.brake_delay = brake_delay
        self.gas_std = gas_std
        self.brake_std = brake_std
        self.delay = []
        self.delay.append(gas_delay)
        self.delay.append(brake_delay)
        self.delay_std = []
        self.delay_std.append(gas_std)
        self.delay_std.append(brake_std)
        self.pre_actions = [1] * self.nagents
        self.car_state = [1] * self.nagents
        self.action_timer = [self.gas_delay] * self.nagents
        # self.action_buffer = [dict() for _ in range(self.nagents)]
        self.t = 0
        # self.action_buffer = [dict() for _ in range(self.env.n_agents)]
        self.action_buffer = np.zeros((self.nagents, self.episode_limit))
        self.valid_buffer = np.zeros((self.nagents, self.episode_limit))
        # np.random.seed(seed)

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''

        # tuple space
        if hasattr(self.env.observation_space, 'spaces'):
            total_obs_dim = 0
            for space in self.env.observation_space.spaces:
                if hasattr(self.env.action_space, 'shape'):
                    total_obs_dim += int(np.prod(space.shape))
                else:  # Discrete
                    total_obs_dim += 1
            return total_obs_dim
        else:
            return int(np.prod(self.env.observation_space.shape))

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            # Discrete
            return self.env.action_space.n

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, actions):
        """ Returns reward, terminated, info """
        # print(actions)
        new_actions = []
        for i, action in enumerate(actions):
            action_delay = 0
            # delay action

            if self.delay_std[action] == 0 and self.delay[action] > 0:
                action_delay = self.delay[action]
            elif self.delay_std[action] > 0 and self.delay[action] > 0:
                action_delay = int(np.random.normal(loc=self.delay[action], scale=self.delay_std[action]))

            if self.t + max(0, action_delay) < self.episode_limit:
                self.action_buffer[i][self.t + max(0, action_delay)] = action
                self.valid_buffer[i][self.t + max(0, action_delay)] = 1
            # self.action_buffer[i][self.t + max(0, action_delay)] = action

            if self.valid_buffer[i][self.t] == 1:
                new_actions.append(self.action_buffer[i][self.t])
                self.valid_buffer[i][self.t] = 0
            else:
                new_actions.append(1)
            # if self.t in self.action_buffer[i]:
            #     new_actions.append(self.action_buffer[i].pop(self.t))
            # else:
            #     new_actions.append(1)

        obs, rewards, dones, _ = self.env.step(new_actions)
        self.obs = self._flatten_obs(obs)

        if self.display:
            self.env.render()

        reward = np.sum(rewards)
        terminated = np.all(dones)

        info = {"battle_won": self.env.stat["success"]}
        self.t += 1

        return obs, reward, terminated, False, info

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs = []
            for agent in obs:  # list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)

        obs = obs.reshape(-1, self.observation_dim)
        return obs

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise np.array(self.obs[agent_id])

    def get_obs_size(self):  # fancp
        """ Returns the shape of the observation """
        return self.observation_dim

    def get_state(self):
        return np.concatenate(self.obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.nagents * self.get_obs_size()

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.nagents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return flatdim(self.env.action_space) * [1]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.num_actions

    def reset(self):
        """ Returns initial observations and states"""
        self.pre_actions = [1] * self.nagents
        self.car_state = [1] * self.nagents
        self.action_timer = [self.gas_delay] * self.nagents
        obs = self.env.reset()
        self.obs = self._flatten_obs(obs)
        self.t = 0

        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        return self.env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        stats = {
            "success": self.env.stat,
        }
        return stats

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.nagents,
                    "episode_limit": self.episode_limit}
        return env_info


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


if __name__ == '__main__':
    env = Traffic_JunctionEnv(add_rate_max=0.2, display=True, seed=222, add_rate_min=0.05, curr_end=0, curr_start=0,
                              difficulty='medium', dim=14, nagents=10, vision=0, vocab_type='bool')
    print(env.get_env_info())

    env.reset()
    print(type(env.get_state()), env.get_state()[0].shape)
    print(len(env.get_obs()), env.get_obs()[0].shape, type(env.get_obs()))

    agent = RandomAgent(env.action_space)

    for i in range(100):
        actions = []
        for _ in range(10):
            action = agent.act()
            actions.append(action)
        actions = torch.from_numpy(np.array(actions))
        obs, reward, _ = env.step(actions)
        print(env.get_stats())