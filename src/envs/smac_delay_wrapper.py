from smac.env import StarCraft2Env
import numpy as np

from .multiagentenv import MultiAgentEnv


class SMACDelayWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, **kwargs):
        self.delay_mean = kwargs['delay_mean']
        self.delay_std = kwargs['delay_std']
        del kwargs['delay_mean']
        del kwargs['delay_std']
        del kwargs['common_reward']
        del kwargs['reward_scalarisation']
        self.env = StarCraft2Env(map_name=map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.action_buffer = [dict() for _ in range(self.env.n_agents)]
        # self.action_buffer = np.zeros((self.env.n_agents, self.episode_limit))
        # self.valid_buffer = np.zeros((self.env.n_agents, self.episode_limit))

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        new_actions = []
        avail_actions = self.get_avail_actions()
        if self.delay_std == 0:
            delay = [self.delay_mean] * self.env.n_agents
        else:
            delay = np.clip(np.random.normal(self.delay_mean, self.delay_std, self.env.n_agents), 0, 2).astype(int)
        for i in range(self.env.n_agents):
            self.action_buffer[i][self.t + delay[i]] = actions[i]
            # if self.t + delay[i] < self.episode_limit:
            #     self.action_buffer[i][self.t + delay[i]] = actions[i]
            #     self.valid_buffer[i][self.t + delay[i]] = 1
        for i in range(self.env.n_agents):
            if self.t in self.action_buffer[i]:
                new_actions.append(self.action_buffer[i].pop(self.t))
            else:
                new_actions.append(avail_actions[i][0] ^ 1)

            # if self.valid_buffer[i][self.t] == 1:
            #     new_actions.append(self.action_buffer[i][self.t])
            #     self.valid_buffer[i][self.t] = 0
            # else:
            #     new_actions.append(avail_actions[i][0] ^ 1)

        for i, action in enumerate(new_actions):
            if avail_actions[i][int(action)] == 0:
                new_actions[i] = avail_actions[i][0] ^ 1
        
        rews, terminated, info = self.env.step(new_actions)
        obss = self.get_obs()
        truncated = False
        self.t += 1
        return obss, rews, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        self.t = 0
        obss, _ = self.env.reset()
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()
