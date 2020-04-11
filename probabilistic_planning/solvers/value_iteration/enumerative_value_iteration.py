def create_value_function_with_default_value(states, default_value):
    tuples = map(lambda state: (state, default_value), states)
    return dict(tuples)

def copy_value_function(value_function):
    states = value_function.keys()
    tuples = map(lambda state: (state, value_function[state]), states)
    return dict(tuples)

def compute_maximum_residual(first_value_function, second_value_function):
    states = first_value_function.keys()
    residuals = map(lambda state: abs(first_value_function[state] - second_value_function[state]), states)
    return max(residuals)

def compute_expected_reward_for_action(state, action, mdp, gamma, value_function):
    pondered_expected_values = []

    for next_state in mdp.states:
        pondered_expected_values.append(
            mdp.transition(state, action, next_state) * value_function[next_state]
        )

    return mdp.reward(state) + gamma * sum(pondered_expected_values)

def compute_bellman_backup(state, mdp, gamma, value_function):
    expected_rewards = []

    for action in mdp.actions:
        expected_rewards.append(
            compute_expected_reward_for_action(state, action, mdp, gamma, value_function)
        )

    return max(expected_rewards)

def compute_policy(mdp, gamma, value_function):
    policy = {}

    for state in mdp.states:
        max_value = float("-inf ")
        max_action = None

        for action in mdp.actions:
            expected_reward = compute_expected_reward_for_action(state, action, mdp, gamma, value_function)

            if expected_reward > max_value:
                max_value = expected_reward
                max_action = action

        policy[state] = max_action

    return policy

def enumerative_value_iteration(mdp, gamma, epsilon, initial_value_function = None):
    """Executes the Value Iteration algorithm.

    Parameters:
    mdp (EnumerativeMDP): enumerative Markov Decison Problem to be solved
    gamma (float): discount factor applied to solve this MDP (assumes infinite on indefinite horizon)
    epsilon (float): maximum residual allowed between V_k and V_{k+1}

    Returns:
    policy (dict): resulting policy computed for a mdp, represented as a dict that maps a state to an action
    value_function (dict): values function found by this algorithm, represented as a dict that maps a state to a float
    iterations (int): number of iterations needed until convergence
    maximum_residuals (list): list of residuals found until convergence
    """
    value_function = initial_value_function

    if value_function is None:
        value_function = create_value_function_with_default_value(states=mdp.states, default_value=0.0)

    iterations = 0
    maximum_residuals = []

    while True:
        computed_value_function = copy_value_function(value_function)

        # do bellman update
        for state in mdp.states:
            computed_value_function[state] = compute_bellman_backup(state, mdp, gamma, value_function)

        iteration_residual = compute_maximum_residual(value_function, computed_value_function)
        value_function = computed_value_function

        # update statistics
        iterations = iterations + 1
        maximum_residuals.append(iteration_residual)

        if iteration_residual < epsilon:
            break # end loop

    # compute policy
    policy = compute_policy(mdp, gamma, value_function)

    return policy, value_function, iterations, maximum_residuals
