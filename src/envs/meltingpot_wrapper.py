# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple human player for testing a Melting Pot level.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to fire the zapper.
Use `TAB` to switch between players.
"""
import queue
import threading
import collections
import enum
import time
import logging
import datetime

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
import dm_env

from ml_collections import config_dict
import numpy as np
import pygame

import dmlab2d
from envs.meltingpot.utils.substrates import builder
from envs.scene_descriptor.scene_descriptor import SceneDescriptor
from envs.scene_descriptor.observations_generator import ObservationsGenerator
from envs.meltingpot.bots import Bot
from envs.meltingpot.utils.utils import default_agent_actions_map, check_agent_out_of_game

from envs.meltingpot.configs.substrates import clean_up

from .multiagentenv import MultiAgentEnv
import sys
import ast

environment_configs = {
    'clean_up': clean_up,
}
env_module = environment_configs['clean_up']
env_config = env_module.get_config(['1', '2', '3', '4', '5', '6', '7'])
with config_dict.ConfigDict(env_config).unlocked() as env_config:
    roles = env_config.default_player_roles
    env_config.lab2d_settings = env_module.build(env_config)

WHITE = (255, 255, 255)

MOVEMENT_MAP = {
    'NONE': 0,
    'FORWARD': 1,
    'RIGHT': 2,
    'BACKWARD': 3,
    'LEFT': 4,
}

EnvBuilder = Callable[..., dmlab2d.Environment]  # Only supporting kwargs.

ActionMap = Mapping[str, Callable[[], int]]


def get_direction_pressed(action) -> int:
    """Gets direction pressed."""
    if action == 1:
        return MOVEMENT_MAP['FORWARD']
    if action == 2:
        return MOVEMENT_MAP['RIGHT']
    if action == 3:
        return MOVEMENT_MAP['BACKWARD']
    if action == 4:
        return MOVEMENT_MAP['LEFT']
    return MOVEMENT_MAP['NONE']


def get_turn_pressed(action) -> int:
    """Calculates turn increment."""
    if action == 5:
        return -1
    if action == 6:
        return 1
    return MOVEMENT_MAP['NONE']


def get_key_number_pressed(action) -> int:
    if action == 7:
        return 1
    return MOVEMENT_MAP['NONE']


_ACTION_MAP = {
    'move': get_direction_pressed,
    'turn': get_turn_pressed,
    'fireClean': get_key_number_pressed,
}


class RenderType(enum.Enum):
    NONE = 0
    PYGAME = 1


def _split_key(key: str) -> Tuple[str, str]:
    """Splits the key into player index and name."""
    return tuple(key.split('.', maxsplit=1))


def _get_rewards(timestep: dm_env.TimeStep) -> Mapping[str, float]:
    """Gets the list of rewards, one for each player."""
    rewards = {}
    for key in timestep.observation.keys():
        if key.endswith('.REWARD'):
            player_prefix, name = _split_key(key)
            if name == 'REWARD':
                rewards[player_prefix] = timestep.observation[key]
    return rewards


class ActionReader(object):
    """Convert keyboard actions to environment actions."""

    def __init__(self, env: dmlab2d.Environment, action_map: ActionMap):
        # Actions are named "<player_prefix>.<action_name>"
        self._action_map = action_map
        self._action_spec = env.action_spec()
        assert isinstance(self._action_spec, dict)
        self._action_names = set()
        for action_key in self._action_spec.keys():
            _, action_name = _split_key(action_key)
            self._action_names.add(action_name)

    def step(self, player_prefix: str) -> Mapping[str, int]:
        """Update the actions of player `player_prefix`."""
        actions = {action_key: 0 for action_key in self._action_spec.keys()}
        for action_name in self._action_names:
            actions[f'{player_prefix}.{action_name}'] = self._action_map[
                action_name]()
        return actions

    def various_agents_step(self, new_action_map, player_prefixes) -> Mapping[str, int]:
        """Update the actions of player `player_prefix`.
        Args:
            new_action_map: A dictionary with the actions of each player. Keys are player prefixes
            player_prefixes: A list with the player prefixes
        Returns:
            A dictionary with the actions of each player. Keys are combination of player indices starting from 1 and action names
        """
        actions = {action_key: 0 for action_key in self._action_spec.keys()}
        for i, player_prefix in enumerate(player_prefixes):
            for action_name in self._action_names:
                actions[f'{i + 1}.{action_name}'] = new_action_map[player_prefix][action_name]
        return actions

    def various_agents_step_new(self, old_action, player_prefixes) -> Mapping[str, int]:
        """Update the actions of player `player_prefix`.
        Args:
            new_action_map: A dictionary with the actions of each player. Keys are player prefixes
            player_prefixes: A list with the player prefixes
        Returns:
            A dictionary with the actions of each player. Keys are combination of player indices starting from 1 and action names
        """
        actions = {action_key: 0 for action_key in self._action_spec.keys()}
        for i, player_prefix in enumerate(player_prefixes):
            if old_action[i] <= 4:
                actions[f'{i + 1}.{"move"}'] = old_action[i]
            elif old_action[i] == 5:
                actions[f'{i + 1}.{"turn"}'] = 1
            elif old_action[i] == 6:
                actions[f'{i + 1}.{"turn"}'] = -1
            elif old_action[i] == 7:
                actions[f'{i + 1}.{"fireClean"}'] = 1
        return actions


class MeltingPotWrapper(MultiAgentEnv):
    """Run multiplayer environment, with per player rendering and actions. This class is used to run the game Commons Harvest Open from Meltingpot."""

    def __init__(self,
                 render_observation: str,
                 config_overrides: Dict[str, Any],
                 game_ascii_map: str,
                 action_map: ActionMap = _ACTION_MAP,
                 full_config: config_dict.ConfigDict = env_config,
                 interactive: RenderType = RenderType.PYGAME,
                 window_size_x: int = 800,
                 window_size_y: int = 600,
                 state_timestep_number: bool = False,
                 seed: int = 0,
                 fps: int = 8,
                 verbose_fn: Optional[Callable[[dm_env.TimeStep, int, int], None]] = None,
                 text_display_fn: Optional[Callable[[dm_env.TimeStep, int], str]] = None,
                 text_font_size: int = 36,
                 text_x_pos: int = 20,
                 text_y_pos: int = 20,
                 text_color: Tuple[int, ...] = WHITE,
                 env_builder: EnvBuilder = builder.builder,
                 print_events: Optional[bool] = False,
                 player_prefixes: Optional[Sequence[str]] = None,
                 default_observation: str = 'WORLD.RGB',
                 reset_env_when_done: bool = False,
                 bots: Optional[list[Bot]] = None,
                 substrate_name: str = 'clean_up',
                 common_reward: bool = False,
                 reward_scalarisation: str = 'mean',
                 time_limit: int = 50,
                 action_possibility: int = 0.1,
                 action_latency: int = 1,
                 **kwargs,
                 ):
        """Run multiplayer environment, with per player rendering and actions.

        This function initialises a Melting Pot environment with the given
        configuration (including possible config overrides), and optionally launches
        the episode as an interactive game using pygame.  The controls are described
        in the action_map, whose keys correspond to discrete actions of the
        environment.

        Args:
        render_observation: A string consisting of the observation name to render.
            Usually 'RGB' for the third person world view.
        config_overrides: A dictionary of settings to override from the original
            `full_config.lab2d_settings`. Typically these are used to set the number
            of players.
        action_map: A dictionary of (discrete) action names to functions that detect
            the keys that correspond to its possible action values.  For example,
            for movement, we might want to have WASD navigation tied to the 'move'
            action name using `get_direction_pressed`.  See examples in the various
            play_*.py scripts.
        full_config: The full configuration for the Melting Pot environment.  These
            usually come from meltingpot/python/configs/environments.
        game_ascii_map: The ascii map of the game.
        interactive: A RenderType representing whether the episode should be run
            with PyGame, or without any interface.  Setting interactive to false
            enables running e.g. a random agent via the action_map returning actions
            without polling PyGame (or any human input).  Non interactive runs
            ignore the screen_width, screen_height and fps parameters.
        screen_width: Width, in pixels, of the window to render the game.
        screen_height: Height, in pixels, of the window to render the game.
        fps: Frames per second of the game.
        verbose_fn: An optional function that will be executed for every step of
            the environment.  It receives the environment timestep, a player index
            (will be called for every index), and the current player index. This is
            typically used to print extra information that would be useful for
            debugging a running episode.
        text_display_fn: An optional function for displaying text on screen. It
            receives the environment and the player index, and returns a string to
            display on the pygame screen.
        text_font_size: the font size of onscreen text (from `text_display_fn`)
        text_x_pos: the x position of onscreen text (from `text_display_fn`)
        text_y_pos: the x position of onscreen text (from `text_display_fn`)
        text_color: RGB color of onscreen text (from `text_display_fn`)
        env_builder: The environment builder function to use. By default it is
            meltingpot.builder.
        print_events: An optional bool that if enabled will print events captured
            from the dmlab2d events API on any timestep where they occur.
        player_prefixes: If given, use these as the prefixes of player actions.
            Pressing TAB will cycle through these. If not given, use the standard
            ('1', '2', ..., numPlayers).
        default_observation: Default observation to render if 'render_observation'
            or '{player_prefix}.{render_observation}' is not found in the dict.
        reset_env_when_done: if True, reset the environment once the episode has
            terminated; useful for playing multiple episodes in a row. Note this
            will cause this function to loop infinitely.
        bots: A list of Bot objects. This bots have a predefined policy.
        substrate_name: The name of the substrate to use. By default it is 'commons_harvest_open'.
        """
        # Update the config with the overrides.
        full_config.lab2d_settings.update(config_overrides)
        # Create a descriptor to get the raw observations from the game environment
        descriptor = SceneDescriptor(full_config, substrate_name)

        # Define the player ids
        if player_prefixes is None:
            player_count = full_config.lab2d_settings.get('numPlayers', 1)
            # By default, we use lua indices (which start at 1) as player prefixes.
            player_prefixes = [f'{i + 1}' for i in range(player_count)]
        else:
            player_count = len(player_prefixes)

        # Create the game environment
        env = env_builder(**full_config)

        # Reset the game environment
        timestep = env.reset()

        # Create a dictionary to store the score of each player
        score = collections.defaultdict(float)

        # Set the pygame variables
        if interactive == RenderType.PYGAME:
            pygame.init()
            pygame.display.set_caption('Melting Pot: {}'.format(
                full_config.lab2d_settings.levelName))
            font = pygame.font.SysFont(None, text_font_size)

        scale = 1
        observation_spec = env.observation_spec()
        if render_observation in observation_spec:
            obs_spec = observation_spec[render_observation]
        elif f'1.{render_observation}' in observation_spec:
            # This assumes all players have the same observation, which is true for
            # MeltingPot environments.
            obs_spec = observation_spec[f'1.{render_observation}']
        else:
            # Falls back to 'default_observation.'
            obs_spec = observation_spec[default_observation]

        observation_shape = obs_spec.shape
        observation_height = observation_shape[0]
        observation_width = observation_shape[1]
        scale = min(window_size_y // observation_height,
                    window_size_x // observation_width)
        if interactive == RenderType.PYGAME:
            game_display = pygame.display.set_mode(
                (observation_width * scale, observation_height * scale))
            clock = pygame.time.Clock()

        self.n_agents = len(player_prefixes)
        self.episode_limit = time_limit

        self.env = env
        self.pygame = pygame

        self.first_move_done = False
        self.interactive = interactive
        self.player_prefixes = player_prefixes
        self.player_count = player_count
        self.action_map = action_map
        self.descriptor = descriptor
        self.timestep = timestep
        self.reset_env_when_done = reset_env_when_done
        self.verbose_fn = verbose_fn
        self.text_display_fn = text_display_fn
        self.font = font
        self.text_font_size = text_font_size
        self.text_x_pos = text_x_pos
        self.text_y_pos = text_y_pos
        self.text_color = text_color
        self.print_events = print_events
        self.default_observation = default_observation
        self.score = score
        self.render_observation = render_observation
        self.scale = scale
        self.screen_width = window_size_x
        self.screen_height = window_size_y
        self.fps = fps
        self.game_display = game_display
        self.clock = clock
        self.observationsGenerator = ObservationsGenerator(game_ascii_map, player_prefixes, substrate_name)
        self.time = datetime.timedelta()
        self.game_steps = 0  # Number of steps of the game
        self.bots = bots
        self.curr_scene_description = None
        self.game_ascii_map = game_ascii_map
        self.substrate_name = substrate_name

        self.action_possibility = action_possibility
        self.action_timer = self.n_agents * [0]
        self.action_recoder = self.n_agents * [0]
        # self.action_map = self.n_agents * [self.get_state_size() * [0]]
        self.latency = action_latency
        self.action_queue = np.zeros((7, 205, 3))

        self.punish_t = self.episode_limit

    def actions_process(self, actions):
        new_actions = []
        action_dict = {1: 101, 2: 114, 3: 125, 4: 112}
        dis_dict = {1: -30, 2: 1, 3: 30, 4: -1}

        for i, action in enumerate(actions):
            tmp_1 = (self.curr_scene_description[str(i + 1)]['global_position'][0]) * 30 + \
                    (self.curr_scene_description[str(i + 1)]['global_position'][1])
            tmp_2 = 0
            tmp_3 = 0
            if 0 < action <= 4:
                t_tmp = (action + self.curr_scene_description[str(i + 1)]['orientation'])
                if t_tmp > 4:
                    t_tmp -= 4
                tmp_2 = (self.curr_scene_description[str(i + 1)]['global_position'][0]) * 30 + \
                        (self.curr_scene_description[str(i + 1)]['global_position'][1]) + dis_dict[t_tmp.item()]
            if (0 < action <= 4) and self.curr_scene_description[str(i + 1)]['observation'][
                action_dict[action.item()]] == 'A' and self.latency > 0:
                tmp_3 = max(0, int(np.random.normal(loc=self.latency, scale=self.action_possibility)))
                tmp_2 *= -1
                self.action_recoder[i] = action
                action = 0

            self.action_queue[i][self.time.days][0] = tmp_1
            self.action_queue[i][self.time.days][1] = tmp_2
            self.action_queue[i][self.time.days][2] = tmp_3
            if tmp_2 < 0 and self.latency > 0:
                for j in range(self.time.days):
                    if (self.action_queue[i][j][0] == self.action_queue[i][self.time.days][0] and
                        self.action_queue[i][j][1] == self.action_queue[i][self.time.days][1]) and \
                            self.time.days - j == self.action_queue[i][j][2]:
                        action = self.action_recoder[i]

            new_actions.append(action)
        return new_actions

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        """Run one step of the game.

                Args:
                    actions: A dictionary of actions for each player.
                Returns:
                    A dictionary with the observations of each player.
                """
        stop = False
        # Check for pygame controls
        if self.interactive == RenderType.PYGAME:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop = True

        if stop:
            return self.get_obs(), 0, True, True, self.get_env_info()

        current_actions_map = {}

        action_reader = ActionReader(self.env, self.action_map)
        # Get the raw observations from the environment
        description, curr_global_map = self.descriptor.describe_scene(self.timestep)
        self.curr_scene_description = description
        self.curr_global_map = curr_global_map
        actions = self.actions_process(actions)

        prev_global_map = self.prev_global_map.copy() if hasattr(self, 'prev_global_map') else None

        if self.first_move_done:
            self.game_steps += 1
            # Get the next action map
            game_actions = action_reader.various_agents_step_new(actions, self.player_prefixes)
            self.timestep = self.env.step(game_actions)

            # Update the time: One hour per step
            self.time += datetime.timedelta(days=1)
        else:
            self.first_move_done = True
        ## --------- END OF OUR CODE ---------

        # Get the rewards
        rewards = _get_rewards(self.timestep)
        reward = 0.0
        for value in rewards.values():
            reward += value

        # Check if the game is finished
        if self.timestep.step_type == dm_env.StepType.LAST:
            if self.reset_env_when_done:
                self.timestep = self.env.reset()
            else:
                return self.get_obs(), reward, True, True, self.get_env_info()

        for i, prefix in enumerate(self.player_prefixes):
            if self.verbose_fn:
                self.verbose_fn(self.timestep, i)
            self.score[prefix] += rewards[str(i + 1)]

        # Print events if applicable
        if self.print_events and hasattr(self.env, 'events'):
            events = self.env.events()

        # pygame display
        if self.interactive == RenderType.PYGAME:
            # show visual observation
            if self.render_observation in self.timestep.observation:
                obs = self.timestep.observation[self.render_observation]
            else:
                # Fall back to default_observation.
                obs = self.timestep.observation[self.default_observation]
            obs = np.transpose(obs, (1, 0, 2))  # PyGame is column major!

            surface = pygame.surfarray.make_surface(obs)
            rect = surface.get_rect()

            surf = pygame.transform.scale(
                surface, (rect[2] * self.scale, rect[3] * self.scale))
            self.game_display.blit(surf, dest=(0, 0))

            # show text
            if self.text_display_fn:
                if self.player_count == 1:
                    text_str = self.text_display_fn(self.timestep, 0)
                else:
                    text_str = self.text_display_fn(self.timestep)
                img = self.font.render(text_str, True, self.text_color)
                self.game_display.blit(img, (self.text_x_pos, self.text_y_pos))

            # tick
            pygame.display.update()
            self.clock.tick(self.fps)

        self.prev_global_map = curr_global_map

        # Get the raw observations from the environment after the actions are executed
        description, curr_global_map = self.descriptor.describe_scene(self.timestep)

        # Update the observations generator
        game_time = self.get_time()
        self.observationsGenerator.update_state_changes(description, game_time)

        self.curr_scene_description = description
        self.curr_global_map = curr_global_map

        a_cnt = 0
        for i in range(len(self.curr_global_map)):
            for j in range(len(self.curr_global_map[i])):
                if self.curr_global_map[i][j] == 'w' and self.prev_global_map[i][j] == 'D':
                    a_cnt += 1

        # return reward - min(self.time.days,
        #                     self.punish_t) * 0.0001 + a_cnt * 0.0002, self.time.days >= self.episode_limit, self.get_env_info()
        return reward + a_cnt * 0.1, self.time.days >= self.episode_limit, self.get_env_info()

    def get_obs(self):
        """Returns all agent observations in a list"""
        all_obs = []
        for i in range(self.n_agents):
            obs = self.get_obs_agent(i)
            all_obs.append(obs)
        return all_obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        # curr_state = \
        #     self.observationsGenerator.get_all_observations_descriptions(str(self.curr_scene_description).strip())[
        #         agent_id]
        scene_description = self.curr_scene_description[str(agent_id + 1)]['observation']
        obs = []
        for word in scene_description:
            if word == 'S':
                code = 1
            elif word == 'w':
                code = 2
            elif word == 'D':
                code = 3
            elif word == 'G':
                code = 4
            elif word == 'A':
                code = 5
            elif word == 'W':
                code = 7
            else:
                code = 6
            if word != '\n':
                # for i in range(7):
                    # if (i + 1) == code:
                    #     obs.append(1)
                    # else:
                    #     obs.append(0)
                obs.append(code)
        return obs

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return 11 * 11

    def get_state(self):
        state = []
        for words in self.curr_global_map:
            for word in words:
                if word == 'S':
                    code = 1
                elif word == 'w':
                    code = 2
                elif word == 'D':
                    code = 3
                elif word == 'G':
                    code = 4
                elif word == 'A':
                    code = 5
                elif word == 'W':
                    code = 7
                else:
                    code = 6
                if word != '\n':
                    # for i in range(7):
                    #     if (i + 1) == code:
                    #         state.append(1)
                    #     else:
                    #         state.append(0)
                    state.append(code)
        return state

    def get_state_size(self):
        """Returns the shape of the state"""
        return 21 * 30

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = 8 * [1]
        return valid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return 8

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        self.first_move_done = False
        self.timestep = self.env.reset()
        self.time = datetime.timedelta()
        self.game_steps = 0
        self.step(self.n_agents * [0])
        self.action_queue = np.zeros((7, 205, 3))

        self.punish_t = self.episode_limit
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        self.env.close()
        self.pygame.quit()

    def seed(self, seed=None):
        pass

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": 200,
        }
        return env_info

    def get_stats(self):
        return {}

    def end_game(self):
        """Ends the game. This function is called when the game is finished."""
        game_steps = self.game_steps
        self.env.close()
        self.env = None
        self.pygame.quit()
        self.pygame = None

    def get_observations_by_player(self, player_prefix: str) -> dict:
        """Returns the observations of the given player.
        Args:
            player_prefix: The prefix of the player
        Returns:
            A dictionary with the observations of the player
        """
        curr_state = \
            self.observationsGenerator.get_all_observations_descriptions(str(self.curr_scene_description).strip())[
                player_prefix]
        scene_description = self.curr_scene_description[player_prefix]
        if (check_agent_out_of_game(curr_state)):
            state_changes = []
        else:
            # When the agent is out, do not get the state changes to accumulate them until the agent is revived
            state_changes = self.observationsGenerator.get_observed_changes_per_agent(player_prefix)
        return {
            'curr_state': curr_state,
            'scene_description': scene_description,
            'state_changes': state_changes
        }

    def get_time(self) -> str:
        """Returns the current time of the game. The time will be formatted as specified in the config file."""
        return f'step {self.time.days}'

    def get_agents_view_imgs(self) -> dict:
        """Returns the images of the agents' views."""
        agents_view_imgs = {}

        for player_id, player_name in enumerate(self.player_prefixes):
            agent_observation = self.timestep.observation[f"{player_id + 1}.RGB"]
            agents_view_imgs[player_name] = agent_observation

        return agents_view_imgs

    def get_agents_orientations(self) -> dict:
        """Returns the orientations of the agents."""
        orientations = {}

        for player_id, player_name in enumerate(self.player_prefixes):
            orientation = self.curr_scene_description[player_name]['orientation']
            orientations[player_name] = orientation
        return orientations

    def get_current_step_number(self) -> int:
        """Returns the current step number of the game."""
        return self.game_steps

    def get_current_global_map(self) -> dict:
        """Returns the current scene description."""
        return self.curr_global_map
