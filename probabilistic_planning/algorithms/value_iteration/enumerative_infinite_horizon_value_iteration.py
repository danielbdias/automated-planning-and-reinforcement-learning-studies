from probabilistic_planning.structures import EnumerativeValueFunction

def compute_maximum_residual(mdp, first_value_function, second_value_function):
    state_residual = lambda state: abs(first_value_function[state] - second_value_function[state])
    residuals = map(state_residual, mdp.states)
    return max(residuals)

def compute_quality(state, action, mdp, gamma, value_function):
    pondered_expected_values = []

    for next_state in mdp.states:
        pondered_expected_values.append(
            mdp.transition(state, action, next_state) * value_function[next_state]
        )

    return mdp.reward(state) + gamma * sum(pondered_expected_values)

def compute_bellman_backup(state, mdp, gamma, value_function):
    qualities = []

    for action in mdp.actions:
        qualities.append(
            compute_quality(state, action, mdp, gamma, value_function)
        )

    return max(qualities)

def compute_policy(mdp, gamma, value_function):
    policy = {}

    for state in mdp.states:
        max_value = float("-inf ")
        best_action = None

        for action in mdp.actions:
            quality = compute_quality(state, action, mdp, gamma, value_function)

            if quality > max_value:
                max_value = quality
                best_action = action

        policy[state] = best_action

    return policy

def enumerative_infinite_horizon_value_iteration(mdp, gamma, epsilon, initial_value_function = None):
    """Executes the Value Iteration algorithm for infinete or indefinite horizon MDPs.

    Parameters:
    mdp (EnumerativeMDP): enumerative Markov Decison Problem to be solved
    gamma (float): discount factor applied to solve this MDP (assumes infinite on indefinite horizon)
    epsilon (float): maximum residual allowed between V_k and V_{k+1}
    initial_value_function (EnumerativeValueFunction): initial value function to start the algorithm. If this value
                                                       is ommited, the algorithm will consider a initial value function that
                                                       that returns 0 (zero) for all states.

    Returns:
    policy (dict): resulting policy computed for a mdp, represented as a dict that maps a state to an action
    value_function (dict): value function found by this algorithm, represented as a dict that maps a state to a float
    statistics (dict): dictionary containing some statistics about the algorithm execution. We have three statistics here:
                      "iterations" that is equal to the horizon parameter, "bellman_backups_done" that is the overall
                      number of Bellman backups executed and "maximum_residuals" that is the maximum residual found in
                      each iteration.
    """
    value_function = initial_value_function

    if value_function is None:
        value_function = EnumerativeValueFunction(lambda state: 0) # value function with zeroes

    iterations = 0
    maximum_residuals = []
    bellman_backups_done = 0

    while True:
        computed_value_function = value_function.copy()

        # do bellman update
        for state in mdp.states:
            computed_value_function[state] = compute_bellman_backup(state, mdp, gamma, value_function)
            bellman_backups_done = bellman_backups_done + 1

        iteration_residual = compute_maximum_residual(mdp, value_function, computed_value_function)
        value_function = computed_value_function

        # update statistics
        iterations = iterations + 1
        maximum_residuals.append(iteration_residual)

        if iteration_residual < epsilon:
            break # end loop

    # compute policy
    policy = compute_policy(mdp, gamma, value_function)

    statistics = {
        "iterations": iterations,
        "maximum_residuals": maximum_residuals,
        "bellman_backups_done": bellman_backups_done
    }

    return policy, value_function, statistics
