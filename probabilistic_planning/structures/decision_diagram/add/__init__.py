"""Structures and helper functions to handle algebraic decision digrams.
"""

# ("c1",
#   (
#       ("c1", ( 1, 0.5 )),
#       ("c1", ( 1, 0.5 ))
#   )
# )

from ....helpers import validate_defined_argument
from .. import ADD

def read_add_from_tuples(tuple_value, state_variables):
    validate_defined_argument(tuple_value, "tuples")
    validate_defined_argument(state_variables, "state_variables")

    return convert_tuple_to_add(tuple_value, state_variables)

def convert_tuple_to_add(add_as_tuple, state_variables, trajectory = []):
    # leaf node on ADD
    if type(add_as_tuple) is float or type(add_as_tuple) is int:
        return ADD.constant(add_as_tuple)

    if type(add_as_tuple) is not tuple:
        raise ValueError("The ADD representation must be a tuple!")

    if len(add_as_tuple) != 2:
        raise ValueError("Invalid tuple ADD representation!")

    # inner node
    state_variable, high_and_low = add_as_tuple

    if type(state_variable) is not str:
        raise ValueError("The inner nodes must have a string and a tuple in ADD representation!")

    if type(high_and_low) is not tuple:
        raise ValueError("The inner nodes must have a string and a tuple in ADD representation!")

    if len(high_and_low) != 2:
        raise ValueError("The inner nodes must have a 2-element tuple in ADD representation!")

    high, low = high_and_low

    state_variable_index = state_variables.index(state_variable)

    if state_variable in trajectory:
        # creates prime variable
        state_variable_index = state_variable_index + len(state_variables)

    trajectory.append(state_variable)

    variable = ADD.variable(state_variable_index)

    # build truthy branch
    high_add = convert_tuple_to_add(high, state_variables, trajectory)
    high_var = variable * high_add

    # build falsy branch
    low_add = convert_tuple_to_add(low, state_variables, trajectory)
    low_var = ~variable * low_add

    # merge both into a single ADD
    return low_var + high_var

__all__ = ["read_add_from_tuples"]
