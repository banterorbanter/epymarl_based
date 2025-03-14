# from agent.memory_structures.short_term_memory import ShortTermMemory
from envs.meltingpot.utils.utils import connected_elems_map, get_element_global_pos, check_agent_out_of_game, get_matrix
import numpy as np
from importlib import import_module

def import_game(substrate_name:str, kind_experiment:str = ""):
    """
    Import the game module from the game_environment/substrates/python folder
    Args:
        substrate_name: Name of the game to import
        kind_experiment: The kind of experiment that will bi run, valid options are: '' for no experiment, 'adversarial_event' for the adversarial event experiment, 'personalized' pre-loaded experiments
    Returns:
        The game module
    """
    substrate_name = f'{substrate_name}___{kind_experiment}' if kind_experiment!="" else substrate_name
    game_module_path = f'game_environment.substrates.python.{substrate_name}'
    game_module = import_module(game_module_path)
    return game_module

def get_scenario_map  (game_name:str)-> str:
    """Get the scenario map from the game environment
    Args:
        game_name: Name of the game to run, the name must match a folder in game_environment/substrates/python
    Returns:
        A string of the scenario map, rows are separed by '\n'
    """
    global ASCII_MAP
    ASCII_MAP = import_game(game_name).ASCII_MAP
    return ASCII_MAP

substrate_name = "clean_up"
scenario_obstacles = {"impassable_obstacles": ["W", "$"] + [str(i) for i in range(10)],
                      "passable_obstacles": ["A"]}

def load_scenario_info(players_context: list[str] = None):
    return {'scenario_map': get_scenario_map(game_name=substrate_name), 
            'valid_actions': get_defined_valid_actions(), 
            'scenario_obstacles': scenario_obstacles}

def default_agent_actions_map():
    """
    Description: Returns the base action map for the agent
    
    Returns:
        dict: Base action map for the agent
    """
    return {
        'move': 0,
        'turn': 0,
        'fireZap': 0,
        'fireClean': 0,
    }


def get_defined_valid_actions():
    """
    Description: Returns the defined valid actions for the substrate
    
    Returns:
        list: List of valid actions that will be given to the agent action module.
    """
    return ['go to position (x,y): This action takes the agent to the position specified, if there is an apple in the position the apple would be taken. You can choose any position on the map from the top left [0, 0] to the bottom right [20, 29]', 
                'immobilize player (player_name) at (x,y): This action takes the agent near the specified position and uses the light beam pointed to the specified position. If there is another agent in that position, the agent would be attacked and will be inactive for a few rounds, then it would be reinstanted on the game on another position.',
                'stay put: This action keep the agent in the same position.',
                'explore: This action makes the agent to explore the map, it moves to a random position on the observed portion of the map.',
                'clean river at (x,y)',
                'go to river bank at (x,y)',
                'go to apples field edge at (x,y)',]
    
    

# def update_known_objects(observations: list[str], stm: ShortTermMemory):
#     """Updates the known agents in the short term memory.
#
#     Args:
#         observations (list[str]): List of observations of the environment.
#         stm (ShortTermMemory): Short term memory of the agent.
#
#     Returns:
#         None
#     """
#     pass # For this substrate theres not need to update the known objects.
#
#
# def get_textually_known_objects(stm: ShortTermMemory):
#     """
#         For this substrate it will return nothing.
#     Args:
#         stm (ShortTermMemory): Short term memory of the agent.
#
#     Returns:
#         str: Textual representation of the known objects.
#         For this substrate it will return nothing.
#     """
#     return ""
    
    
    
def condition_to_end_game( current_map:list[str]):
    """
    Check if the game has ended
    Args:
        current_map: The current map of the game
    Returns:
        A boolean indicating if the game has ended if condition for the specific substrate is met
    """
    # For Clean Up we dont have a condition to end the game
    return False


def get_connected_elements(global_map: str):
    """
    Get the connected elements of the map
    
    Args:
        global_map (str): Global map of the game
        
    Returns:
        dict: Dictionary with the connected elements
    """
    
    connected_elements = {}
    connected_elements['river_bank'] = connected_elems_map(global_map,  ['=','+'])
    connected_elements['apple_field_edge'] = connected_elems_map(global_map, ['^','T'])
    return connected_elements
    
    
    
    
def get_specific_substrate_obs(local_map:str, local_position:tuple, global_position:tuple, agent_orientation:int, connected_elements:dict, symbols:dict):
        
    """
    Description: Returns a list with the descriptions of the objects observed by the agent

    Args:
        local_map (str): Local map in ascci format
        local_position (tuple): Local position of the agent
        global_position (tuple): Global position of the agent
        agent_orientation (int): Orientation of the agent
        
    Returns:
        list: List with the descriptions of the objects observed by the agent
    """
    
    items_observed = []
    # Get apples (A) and dirt (D) observed descriptions
    for i, row in enumerate(local_map.split('\n')):
        for j, char in enumerate(row):
            if char == 'A':
                apple_global_pos = get_element_global_pos((i,j), local_position, global_position, agent_orientation)
                items_observed.append("Observed an apple at position {}".format(apple_global_pos))

            elif char == 'D':
                dirt_global_pos = get_element_global_pos((i,j), local_position, global_position, agent_orientation)
                items_observed.append("Observed dirt on the river at position {}".format(dirt_global_pos))

            for elm in connected_elements['river_bank'].values():
                if (i,j) in elm['elements']:
                    river_bank_global_pos = get_element_global_pos((i,j), local_position, global_position, agent_orientation)
                    items_observed.append("Observed river bank at position {}".format(river_bank_global_pos))
            
            for elm in connected_elements['apple_field_edge'].values():
                if (i,j) in elm['elements']:
                    apple_field_edge_global_pos = get_element_global_pos((i,j), local_position, global_position, agent_orientation)
                    items_observed.append("Observed apple field edge at position {}".format(apple_field_edge_global_pos))

    return items_observed
    




def get_observed_changes(observed_map: str, last_observed_map: str | None, agent_local_position: tuple, agent_global_position: tuple, agent_orientation: int, game_time: str, players_names: dict, agent_name:str) -> list[tuple[str, str]]:
    """Create a list of observations of the changes in the environment
    
    Args:
        observed_map (str): Map observed by the agent
        last_observed_map (str | None): Last map observed by the agent
        agent_local_position (tuple): Position of the agent on the observed map
        agent_global_position (tuple): Global position of the agent
        agent_orientation (int): Orientation of the agent
        game_time (str): Current game time

    Returns:
        list[tuple[str, str]]: List of tuples with the changes in the environment, and the game time
    """
    if check_agent_out_of_game([observed_map]):
        return [(observed_map, game_time)]
    
    observations = []
    if last_observed_map == None:
        return observations
    
    curr_m = get_matrix(observed_map)
    last_m = get_matrix(last_observed_map)
    for index in np.ndindex(curr_m.shape):
        curr_el = curr_m[index]
        last_el = last_m[index]
        if curr_el != last_el:
            # If someone attacked nearby
            if curr_el == 'S':
                continue
            if last_el.isnumeric() and curr_el == 'B':
                el_pos = get_element_global_pos(index, agent_local_position, agent_global_position, agent_orientation)
                observations.append((f"Someone was attacked at position {el_pos}.", game_time))
            elif curr_el == 'B':
                el_pos = get_element_global_pos(index, agent_local_position, agent_global_position, agent_orientation)
                observations.append((f"Observed a ray beam from an attack at position {el_pos}.", game_time))
            elif last_el == 'B':
                pass
            # If an apple was taken
            elif last_el == 'A':
                if not curr_el.isdigit():
                    continue
                agent_id = int(curr_el)
                agent_name = players_names[agent_id]
                el_pos = get_element_global_pos(index, agent_local_position, agent_global_position, agent_orientation)
                observations.append((f"Observed that agent {agent_name} took an apple from position {el_pos}.", game_time))
            # If grass desappeared
            elif last_el == 'G' and curr_el == 'F':
                el_pos = get_element_global_pos(index, agent_local_position, agent_global_position, agent_orientation)
                observations.append((f"Observed that the grass at position {el_pos} disappeared.", game_time))
            # If grass appeared
            elif last_el == 'F' and curr_el == 'G':
                el_pos = get_element_global_pos(index, agent_local_position, agent_global_position, agent_orientation)
                observations.append((f"Observed that grass to grow apples appeared at position {el_pos}.", game_time))
            # If an apple appeared
            elif last_el == 'G' and curr_el == 'A':
                el_pos = get_element_global_pos(index, agent_local_position, agent_global_position, agent_orientation)
                observations.append((f"Observed that an apple grew at position {el_pos}.", game_time))

    return observations
