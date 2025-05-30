from envs.meltingpot.scenario import get_config
from envs.meltingpot.utils.policies.policy import Policy
from envs.meltingpot.bot import build
import dm_env
import logging

def get_bots_for_scenario(scenario: str) -> list[str]:
    """Get the bots for the scenario
    
    Args:
        scenario: Name of the scenario

    Returns:
        A list the policies names for the bots
    """
    try:
        scenario_config = get_config(scenario)
    except KeyError:
        raise KeyError('Scenario not found')
    # Get the bots for the scenario (non-focal players)
    bots_roles = [role for i, role in enumerate(scenario_config.roles) if not scenario_config.is_focal[i]]

    # Find the names of the trained policies for the bots
    bots_names = []

    for i, role in enumerate(bots_roles):
        index = i % len(list(scenario_config.bots_by_role[role]))
        available_names = sorted(list(scenario_config.bots_by_role[role]))
        try:
            bots_names.append(available_names[index])
        except:
            continue #TODO: For now, this alternative avoids scenarios with problems with the number of bots and the roles


    return bots_names
    

class Bot:
    """Bot class to initialize the bot policies and get the actions for the bots
    """
    def __init__(self, policy_name: str, name: str, player_index: int, ACTION_SET: list[dict]):
        """Initialize the bot policy and the player name
        
        Args:
            policy_name: Name of the policy for the bot
            name: Name of the player
            player_index: Index of the player in the game
            ACTION_SET: List of actions for the game
        """
        self.policy = build(policy_name)
        self.name = name
        self.state = self.policy.initial_state()
        self.player_index = player_index
        self.ACTION_SET = ACTION_SET

    def move(self, timestep) -> dict:
        """Get the action for the bot
        
        Args:
            timestep: Current timestep of the game

        Returns:
            The action for the bot
        """
        bot_timestep = dm_env.TimeStep(
            step_type=timestep.step_type,
            reward=timestep.observation[f'{self.player_index}.REWARD'],
            discount=timestep.discount,
            observation={
                'RGB': timestep.observation[f'{self.player_index}.RGB'],
                'ORIENTATION': timestep.observation[f'{self.player_index}.ORIENTATION'],
                'READY_TO_SHOOT': timestep.observation[f'{self.player_index}.READY_TO_SHOOT'],
                'POSITION': timestep.observation[f'{self.player_index}.POSITION'],
                'WORLD.RGB': timestep.observation['WORLD.RGB'],
            }
        )
        action, state = self.policy.step(bot_timestep, self.state)
        self.state = state
        new_action = self.ACTION_SET[action]
        return new_action

