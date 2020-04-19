import numpy as np

from ...helpers import validate_defined_argument

def find_state_index(state_list, state):
    try:
        index = state_list.index(state)
        return index
    except ValueError:
        raise ValueError(f"State [{state}] not found in state set")

def build_transition_add(state_variables_list, transition_function, action):
    transition_matrix_as_dict = transition_function[action]

    number_of_state_variables = len(state_variables_list)
    # transition_matrix = np.zeros((number_of_states, number_of_states))

    # state_transitions = transition_matrix_as_dict.keys()

    # if len(state_transitions) == 0:
    #     raise ValueError(f"The action [{action}] should have at least one transition defined")

    # for state_tuple in state_transitions:
    #     from_state, to_state = state_tuple
    #     transition_probability = transition_matrix_as_dict[state_tuple]

    #     from_state_index = find_state_index(state_list, from_state)
    #     to_state_index = find_state_index(state_list, to_state)

    #     transition_matrix[from_state_index][to_state_index] = transition_probability

    # validate_state_transition_probability_distribution(transition_matrix, state_list, action)

    return transition_matrix

def validate_state_transition_probability_distribution(transition_matrix, state_list, action):
    for from_state in state_list:
        from_state_index = find_state_index(state_list, from_state)
        probability_sum = sum(transition_matrix[from_state_index])

        if probability_sum == 1.0:
            continue

        raise ValueError(
            (f"Invalid probability distribution on [{from_state}] transition"
             f" in action [{action}]. The sum of all transitions probabilities"
             " from this state to others must be 1 (one)")
        )

def build_transition_add_per_action(transition_function, actions, state_variables):
    """Build and validate a transition function."""

    validate_defined_argument(transition_function, "transition function")

    transition_function_actions = transition_function.keys()

    if len(transition_function_actions) == 0:
        raise ValueError("The transition function must have at least one action transition ADD")

    if len(transition_function_actions) != len(actions):
        raise ValueError("The transition function must have one transition ADD per action")

    transition_add_per_action = {}

    for action in transition_function_actions:
        if action not in actions:
            raise ValueError(f"The {action} action is not defined in action list")

        transition_matrix = build_transition_add(state_variables, transition_function, action)
        transition_add_per_action[action] = transition_matrix

    return transition_add_per_action

def build_sorted_list(list_value, list_name):
    validate_defined_argument(list_value, list_name)
    return list(sorted(list_value))

class FactoredTransitionFunction:
    def __init__(self, transition_function, actions, state_variables):
        self.state_variables = build_sorted_list(state_variables, "state variables")
        self.actions = build_sorted_list(actions, "actions")
        self.transition_add_per_action = build_transition_add_per_action(transition_function, self.actions, self.state_variables)

    def get_transition_probability(self, from_state, action, to_state):
        if action not in self.actions:
            raise ValueError(f"Action [{action}] not found")

        transition_matrix = self.transition_matrix_per_action[action]

        from_state_index = find_state_index(self.states, from_state)
        to_state_index = find_state_index(self.states, to_state)

        return transition_matrix[from_state_index][to_state_index]

    def get_transition_matrix(self, action):
        if action not in self.actions:
            raise ValueError(f"Action [{action}] not found")

        transition_matrix = self.transition_matrix_per_action[action]
        return np.matrix(transition_matrix)
