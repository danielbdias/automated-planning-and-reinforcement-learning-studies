"""Module with classes to support a structure that represents
   a Markov Decision Process in Probabilistic Planning."""

from .enumerative_transition_function import EnumerativeTransitionFunction
from ...helpers import validate_defined_argument

import numpy as np

def build_state_list(state_indentifiers, state_list_name, base_state_list=None):
    """Build and validate a state set."""

    validate_defined_argument(state_indentifiers, state_list_name)

    if len(state_indentifiers) == 0:
        raise ValueError(f"The {state_list_name} should have at least one state")

    if len(state_indentifiers) != len(state_indentifiers):
        raise ValueError(f"There is a repeated state identifier in the {state_list_name}")

    if base_state_list is not None:
        for state in state_indentifiers:
            if state not in base_state_list:
                raise ValueError(f"Unrecognized state [{state}] in {state_list_name}")

    return sorted(state_indentifiers)

def build_reward_function(reward_function, states):
    """Build and validate a reward function."""

    validate_defined_argument(reward_function, "reward function")

    if len(reward_function) != len(states):
        raise ValueError("The reward function must be have a value for each state")

    for state in reward_function.keys():
        if state not in states:
            raise ValueError(f"Invalid state [{state}] defined in reward function")

    return reward_function

def build_transition_funtion(transition_function, states):
    """Build and validate a transition function."""

    validate_defined_argument(transition_function, "transition function")

    return EnumerativeTransitionFunction(transition_function, transition_function.keys(), states)

def build_action_list(actions):
    """Build and validate an action set."""

    # transition function already validates the actions
    return sorted(actions)

class EnumerativeMDP:
    """Represents an enumerative Markov Decision Process.

    Attributes:
        states (set): states that are modelled in this MDP
        reward_function (dict): maps a state to its numeric reward
        actions (set): actions that are modelled in this MDP
        transition_function (`TransitionFunction`): an object that given an origin state, an action and a destination states returns the transition
                                                    probability from origin state to destination state
        initial_states (set): (optional) initial states modelled in this MDP
        goal_states (set): (optional) goal states modelled in this MDP
    """

    def __init__(self, states, reward_function, transition_function,
                 initial_states=None, goal_states=None):
        """Initializes a new representation of an enumerative Markov Decision Process.

        Parameters:
            states (list): named states for this MDP
            reward_function (dict): maps named states to the respective numeric reward
            transition_function (dict): maps an action string to another dict that maps a tuple of origin and destination states to a transition probability
            initial_states (list): named states considered initial states for this MDP
            goal_states (list): named states considered goal states for this MDP

        Returns:
            instance (EnumerativeMDP): an enumerative Markov Decision Process
        """
        self.states = build_state_list(states, "states")
        self.reward_function = build_reward_function(reward_function, self.states)
        self.transition_function = build_transition_funtion(transition_function, self.states)
        self.actions = build_action_list(self.transition_function.actions)

        if initial_states:
            self.initial_states = build_state_list(initial_states, "initial states", self.states)
        else:
            self.initial_states = list()

        if goal_states:
            self.goal_states = build_state_list(goal_states, "goal states", self.states)
        else:
            self.goal_states = list()

    def reward(self, state):
        return self.reward_function[state]

    def transition(self, from_state, action, to_state):
        return self.transition_function.get_transition_probability(from_state, action, to_state)

    def reward_matrix(self):
        return np.matrix(list(map(lambda state: [ self.reward(state) ], self.states)))

    def transition_matrix(self, action):
        return self.transition_function.get_transition_matrix(action)