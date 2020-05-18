from probabilistic_planning.structures import EnumerativeValueFunction

import numpy as np

def compute_quality(state, action, mdp, gamma, value_function):
    pondered_expected_values = []

    for next_state in mdp.states:
        pondered_expected_values.append(
            mdp.transition(state, action, next_state) * value_function[next_state]
        )

    return mdp.reward(state) + gamma * sum(pondered_expected_values)

def get_transition_matrix_for_policy(mdp, policy):
    state_count = len(mdp.states)
    transition_matrix_for_policy = np.matrix(np.zeros((state_count, state_count)))

    for index, state in enumerate(mdp.states):
        action = policy[state]
        transition_matrix = mdp.transition_matrix(action)
        transition_matrix_for_policy[index] = transition_matrix[index]

    return transition_matrix_for_policy

def evaluate_policy(policy, mdp, gamma):
    policy_reward_matrix = mdp.reward_matrix()
    policy_transition_matrix = get_transition_matrix_for_policy(mdp, policy)
    identity_matrix = np.identity(len(mdp.states))

    # compute value function as a system of equations
    value_function_matrix_for_policy = (identity_matrix - gamma * policy_transition_matrix).getI() * policy_reward_matrix

    computed_value_function = EnumerativeValueFunction(mdp.states)

    for index, state in enumerate(mdp.states):
        computed_value_function[state] = value_function_matrix_for_policy[index]

    return computed_value_function

def improve_policy(policy, mdp, gamma, value_function):
    improved_policy = dict(policy)

    for state in mdp.states:
        quality_for_best_action = float("-inf ")
        best_action = None

        for action in mdp.actions:
            quality = compute_quality(state, action, mdp, gamma, value_function)

            if quality > quality_for_best_action:
                quality_for_best_action = quality
                best_action = action

        current_action = policy[state]
        quality_for_current_action = compute_quality(state, current_action, mdp, gamma, value_function)

        if quality_for_best_action > quality_for_current_action:
            improved_policy[state] = best_action

    return improved_policy

def equal_policies(mdp, first_policy, second_policy):
    for state in mdp.states:
        first_action = first_policy[state]
        second_action = second_policy[state]

        if first_action != second_action:
            return False

    return True

def enumerative_policy_iteration(mdp, gamma, initial_policy = None):
    """Executes the Policy Iteration algorithm.

    Parameters:
    mdp (EnumerativeMDP): enumerative Markov Decison Problem to be solved
    gamma (float): discount factor applied to solve this MDP
    initial_policy (dict): arbitrary initial policy to start

    Returns:
    policy (dict): resulting policy computed for a mdp, represented as a dict that maps a state to an action
    value_function (dict): value function found by this algorithm, represented as a dict that maps a state to a float
    statistics (dict): dictionary containing some statistics about the algorithm execution. We have one statistic here:
                      "iterations" that is equal to the number of iterations required to improve to the optimal policy.
    """
    policy = initial_policy

    if policy is None:
        first_action = mdp.actions[0]
        policy = dict(map(lambda state: (state, first_action), mdp.states))

    iterations = 0

    while True:
        iterations = iterations + 1

        value_function_for_policy = evaluate_policy(policy, mdp, gamma)
        improved_policy = improve_policy(policy, mdp, gamma, value_function_for_policy)

        if equal_policies(mdp, policy, improved_policy):
            break

        policy = improved_policy

    value_function = evaluate_policy(policy, mdp, gamma)

    statistics = {
        "iterations": iterations
    }

    return policy, value_function, statistics
