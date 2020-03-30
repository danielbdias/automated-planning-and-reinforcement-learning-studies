"""Module with classes to support a structure that represents
   a Markov Decision Process in Probabilistic Planning."""

from .transition_function import TransitionFunction

def validate_defined_argument(argument_value, argument_name):
    """Validates if a given argument has a defined value (is not None)."""
    if argument_value is None:
        raise ValueError(f"The {argument_name} should be defined")

def build_state_set(state_indentifiers, state_set_name, base_state_set=None):
    """Build and validate a state set."""

    validate_defined_argument(state_indentifiers, state_set_name)

    if len(state_indentifiers) == 0:
        raise ValueError(f"The {state_set_name} should have at least one state")

    state_as_set = set(state_indentifiers)
    if len(state_as_set) != len(state_indentifiers):
        raise ValueError(f"There is a repeated state identifier in the {state_set_name}")

    if base_state_set is not None:
        for state in state_indentifiers:
            if state not in base_state_set:
                raise ValueError(f"Unrecognized state [{state}] in {state_set_name}")

    return state_as_set

def build_reward_function(reward_function, states):
    """Build and validate a reward function."""

    validate_defined_argument(reward_function, "reward function")

    if len(reward_function) != len(states):
        raise ValueError("The reward function must be have a value for each state")

    for state in reward_function.keys():
        if state not in states:
            raise ValueError(f"Invalid state [{state}] defined in reward function")

    return reward_function

class EnumerativeMDP:
    """Represents a enumerative Markov Decision Process."""

    def __init__(self, states, reward_function, transition_function,
                 initial_states=None, goal_states=None):
        self.states = build_state_set(states, "states")
        self.reward_function = build_reward_function(reward_function, self.states)
        self.transition_function = TransitionFunction(transition_function, self.states)

        if initial_states:
            self.initial_states = build_state_set(initial_states, "initial states", self.states)
        else:
            self.initial_states = set()

        if goal_states:
            self.goal_states = build_state_set(goal_states, "goal states", self.states)
        else:
            self.goal_states = set()
