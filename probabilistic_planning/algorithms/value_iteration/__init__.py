from .enumerative_infinite_horizon_value_iteration import enumerative_infinite_horizon_value_iteration
from .enumerative_finite_horizon_value_iteration import enumerative_finite_horizon_value_iteration

def enumerative_value_iteration(mdp, **parameters):
    parameter_names = set(parameters.keys())

    if { "gamma", "horizon" } == parameter_names:
        gamma = parameters["gamma"]
        horizon = parameters["horizon"]
        return enumerative_finite_horizon_value_iteration(mdp, gamma, horizon)

    if { "gamma", "epsilon" }.issubset(parameter_names):
        gamma = parameters["gamma"]
        epsilon = parameters["epsilon"]
        initial_value_function = parameters.get("initial_value_function", None)
        return enumerative_infinite_horizon_value_iteration(mdp, gamma, epsilon, initial_value_function)

    raise ValueError(
        ("Invalid parameter configuration."
         " Should receive a gamma and a horizon to run in finite horizon mode or"
         " a gamma, an epsion and a optional initial value function to run on infinite horizon mode.")
    )

__all__ = ["enumerative_value_iteration"]
