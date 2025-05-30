from importlib import import_module
import re
import numpy as np
import logging
from envs.meltingpot.utils.utils import parse_string_to_matrix, matrix_to_string

substrate_utils = None

class Avatar:
    def __init__(self, name:str, avatar_config):
        """
        Avatar class to store information about the player in the game

        Args:
            name (str): Name of the player
            avatar_config (dict): Avatar configuration
        """
        self.name = name
        
        self.avatar_component = list(filter(lambda component: component["component"] == "Avatar",
                                            avatar_config["components"]))[0]
        self.avatar_view = self.avatar_component["kwargs"]["view"]

        self.position = None
        self.orientation = None
        self.last_position = None
        self.last_orientation = None
        self.reward = None
        self.partial_observation = None
        self.last_partial_observation = None
        self.agents_in_observation = None
        self.murder = None
        self.avatar_state = 1
        self.just_died = False
        self.just_revived = False
        self.is_movement_allowed = True


    def set_agents_in_observation(self, agents):
        self.agents_in_observation = agents

    def set_state(self, avatar_state):
        # If the avatar has reappear, set the just_revived flag and reset the murderer name
        if avatar_state == 1 and self.avatar_state == 0:
            self.murder = None
            self.just_revived = True
        # If the avatar just died, set the just_died attribute
        elif avatar_state == 0 and self.avatar_state == 1:
            self. just_died = True
        else:
            self.just_died = False
            self.just_revived = False
        self.avatar_state = avatar_state
    
    def set_is_movement_allowed(self, is_movement_allowed):
        self.is_movement_allowed = is_movement_allowed

    def set_murder(self, murder):
        self.murder = murder

    def set_partial_observation(self, partial_observation):
        self.partial_observation = partial_observation

    def set_last_partial_observation(self, partial_observation):
        self.last_partial_observation = partial_observation

    def set_position(self, x, y):
        self.position = (x, y)

    def set_orientation(self, orientation):
        self.orientation = orientation

    def set_last_position(self):
        self.last_position = self.position

    def set_last_orientation(self):
        self.last_orientation = self.orientation

    def set_reward(self, reward):
        self.reward = reward

    def reset_observation_variables(self):
        self.position = None
        self.orientation = None
        self.reward = None
        self.partial_observation = None
        self.agents_in_observation = None
    
    def __str__(self):
        return (f"Avatar(name={self.name}, view={self.avatar_view}, position={self.position}, "
                f"orientation={self.orientation}, reward={self.reward}, "
                f"partial_observation={self.partial_observation}, "
                f"agents_in_observation={self.agents_in_observation}, murder={self.murder}, "
                f"state={self.avatar_state})")


class SceneDescriptor:

    def __init__(self, substrate_config, substrate_name):
        self.substrate_config = substrate_config
        self.n_players = substrate_config.lab2d_settings.numPlayers
        self.avatars = self.get_avatars(substrate_config.player_names)
        self.last_map = None # Map of the inmediately last step
        global substrate_utils 
        substrate_utils = import_module(f'envs.meltingpot.utilities.{substrate_name}.substrate_utils')
        
        
    def get_avatars(self, names):
        avatars = {}
        # Avatar objects have the value related to the key 'name' equals to 'avatar'. 
        # We look in the self.substrate_config.lab2d_settings.simulation.gameObjects list the ones that have those
        
        avatar_regex = re.compile(r'^avatar\d*$')
        avatar_objects = [game_object for game_object in self.substrate_config.lab2d_settings.simulation.gameObjects if avatar_regex.match(game_object['name'])]
        for i, config in enumerate(avatar_objects):
            avatars[i] = Avatar(str(i + 1), config)
        return avatars

    def describe_scene(self, timestep):
        self.reset_population()
        map, zaps = self.parse_timestep(timestep)
        self.parse_zaps(zaps)
        self.compute_partial_observations(map, self.last_map)
        self.last_map = map
        result = {}
        for avatar_id, avatar in self.avatars.items():
            result[avatar.name] = {"observation": avatar.partial_observation,
                                   "agents_in_observation": avatar.agents_in_observation,
                                   "global_position": avatar.position,
                                   "orientation": int(avatar.orientation),
                                   "local_position": (
                                   avatar.avatar_view.get("forward"), avatar.avatar_view.get("left")),
                                   "last_global_position": avatar.last_position,
                                   "last_orientation": int(
                                       avatar.last_orientation) if avatar.last_orientation is not None else None,
                                   "last_observation": avatar.last_partial_observation,
                                   "effective_zap": avatar.name in [a.murder for a in self.avatars.values() if
                                                                    a.just_died],
                                   "is_movement_allowed": avatar.is_movement_allowed
                                   }
        return result, map

    def parse_zaps(self, zaps):
        for victim_index, row in enumerate(zaps):
            for murder_index, value in enumerate(row):
                murder_name = self.avatars[murder_index].name
                if value > 0:
                    self.avatars[victim_index].set_murder(murder_name)

    def compute_partial_observations(self, map, last_map):
        for avatar_id, avatar in self.avatars.items():
            if avatar.avatar_state == 0:
                if avatar.just_died:
                    obs_text = f"There were no observations because you were attacked by agent {avatar.murder} and you were left out of the game."
                else:
                    obs_text = "There were no observations because you were out of the game."
                avatar.set_partial_observation(obs_text)
                avatar.set_agents_in_observation({})
            else:
                min_padding = max(avatar.avatar_view.values())
                padded_map = self.pad_matrix_to_square(map, min_padding)
                padded_map = np.rot90(padded_map, k=int(avatar.orientation))
                observation, agents_in_observation = self.crop_observation(padded_map, avatar_id, avatar.avatar_view)
                avatar.set_partial_observation(observation)
                avatar.set_agents_in_observation(agents_in_observation)

                # Get the past observations of the observed map to calculate state changes
                if last_map is not None and not avatar.just_revived:
                    last_padded_map = self.pad_matrix_to_square(last_map, min_padding)
                    last_padded_map = np.rot90(last_padded_map, k=int(avatar.last_orientation))
                    last_observation, _ = self.crop_observation(last_padded_map, avatar_id, avatar.avatar_view)
                    avatar.set_last_partial_observation(last_observation)
                # If the avatar just revived, set the last observation to None
                elif last_map is not None and avatar.just_revived:
                    avatar.set_last_partial_observation(None)

    def crop_observation(self, map, avatar_id, avatar_view):
        # get avatar position in matrix
        avatar_pos = np.where(map == str(avatar_id))
        avatar_pos = list(zip(avatar_pos[1], avatar_pos[0]))[0]
        upper_bound = avatar_pos[1] - avatar_view.get("forward")
        left_bound = avatar_pos[0] - avatar_view.get("left")
        lower_bound = avatar_pos[1] + avatar_view.get("backward") + 1
        right_bound = avatar_pos[0] + avatar_view.get("right") + 1
        observation = matrix_to_string(map[upper_bound:lower_bound, left_bound:right_bound])
        observation = observation.replace(str(avatar_id), "#")
        agents_in_observation = self.get_agents_in_observation(observation)
        return observation, agents_in_observation

    def get_agents_in_observation(self, observation):
        digits_list = []
        pattern = r'\d+'

        for string in observation:
            digits = re.findall(pattern, string)
            digits_list.extend(digits)

        agents = {}
        for digit in digits_list:
            agents[digit] = self.avatars[int(digit)].name

        return agents


    @staticmethod
    def pad_matrix_to_square(matrix, min_padding, padding_char="-"):
        num_rows, num_cols = matrix.shape

        max_dim = max(num_rows, num_cols)
        padding_needed = max_dim - min(num_rows, num_cols)
        total_padding = max(padding_needed, min_padding)

        new_dim = max_dim + 2 * total_padding

        padded_matrix = np.full((new_dim, new_dim), padding_char, dtype=matrix.dtype)

        start_row = total_padding
        start_col = total_padding
        padded_matrix[start_row:start_row + num_rows, start_col:start_col + num_cols] = matrix

        return padded_matrix

    def reset_population(self):
        for avatar_id, avatar in self.avatars.items():
            avatar.set_last_position()
            avatar.set_last_orientation()
            avatar.reset_observation_variables()

    def parse_timestep(self, timestep):
        
        map = timestep.observation["GLOBAL.TEXT"].item().decode("utf-8")
        map = parse_string_to_matrix(map)
        # Execute the modify_map_with_rgb function if it exists
        map = substrate_utils.modify_map_with_rgb(map, timestep.observation["WORLD.RGB"]) if hasattr(substrate_utils, 'modify_map_with_rgb') else map
        
        zaps = timestep.observation["WORLD.WHO_ZAPPED_WHO"]
        states = timestep.observation["WORLD.AVATAR_STATES"]
        movement_states = timestep.observation["WORLD.AVATAR_MOVEMENT_STATES"] if "WORLD.AVATAR_MOVEMENT_STATES" in timestep.observation else None
        for avatar_id, avatar in self.avatars.items():
            _id = avatar_id + 1
            position = timestep.observation[f"{_id}.POSITION"]
            if states[avatar_id]: # Only include the avatar in the map if it is alive
                map[position[1], position[0]] = avatar_id
            avatar.set_position(position[1], position[0])
            avatar.set_orientation(timestep.observation[f"{_id}.ORIENTATION"])
            avatar.set_reward(timestep.observation[f"{_id}.REWARD"])
            avatar.set_state(states[avatar_id])
            if movement_states is not None:
                avatar.set_is_movement_allowed(movement_states[avatar_id])

        return map, zaps
    
