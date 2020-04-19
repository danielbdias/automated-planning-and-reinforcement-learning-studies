from ..context import probabilistic_planning
from probabilistic_planning.structures.factored.factored_transition_function import FactoredTransitionFunction

import unittest

class FactoredTransitionFunctionTests(unittest.TestCase):

    # Constructor tests

    def test_constructor_call_with_all_parameters_none(self):
        with self.assertRaisesRegex(ValueError, "The state variables should be defined"):
            FactoredTransitionFunction(transition_function=None, actions=None, state_variables=None)

    def test_constructor_call_with_state_variables_ommited(self):
        transition_function = {
            "some-action": {
                "c1": ("c1", ( ("c1", ( 1, 0.5 )), ("c1", ( 1, 0.5 )) ) )
            }
        }
        actions = {"some-action"}

        with self.assertRaisesRegex(ValueError, "The state variables should be defined"):
            FactoredTransitionFunction(transition_function=transition_function, actions=actions, state_variables=None)

    def test_constructor_call_with_transition_function_ommited(self):
        actions = {"some-action"}
        state_variables = {"state-var-01"}

        with self.assertRaisesRegex(ValueError, "The transition function should be defined"):
            FactoredTransitionFunction(transition_function=None, actions=actions, state_variables=state_variables)

    def test_constructor_call_with_transition_function_without_actions(self):
        transition_function = {}
        actions = {}
        state_variables = {"state-var-01"}

        with self.assertRaisesRegex(ValueError, "The transition function must have at least one action transition ADD"):
            FactoredTransitionFunction(transition_function=transition_function, actions=actions, state_variables=state_variables)

    def test_constructor_call_with_transition_function_having_incorrect_number_of_actions(self):
        transition_function = {
            "some-action": ()
        }
        actions = {"some-action01", "some-action02"}
        state_variables = {"state-var-01"}

        with self.assertRaisesRegex(ValueError, "The transition function must have one transition ADD per action"):
            FactoredTransitionFunction(transition_function=transition_function, actions=actions, state_variables=state_variables)

    def test_constructor_call_with_transition_function_having_undefined_action(self):
        transition_function = {
            "some-weird-action": {}
        }
        actions = {"some-action"}
        state_variables = {"state-var-01"}

        with self.assertRaisesRegex(ValueError, "The some-weird-action action is not defined in action list"):
            FactoredTransitionFunction(transition_function=transition_function, actions=actions, state_variables=state_variables)

    def test_constructor_call_with_transition_function_having_action_with_no_transitions(self):
        transition_function = {
            "some-action": {}
        }
        actions = {"some-action"}
        state_variables = {"state-var-01"}

        with self.assertRaisesRegex(ValueError, "The action \[some-action\] should have at least one transition defined"):
            FactoredTransitionFunction(transition_function=transition_function, actions=actions, state_variables=state_variables)

#     def test_constructor_call_with_transition_function_having_action_with_transition_to_an_invalid_state(self):
#         transition_function_with_invalid_to_state = {
#             "some-action": {
#                 ("invalid-state01", "state01"): 1.0
#             }
#         }
#         transition_function_with_invalid_from_state = {
#             "some-action": {
#                 ("state01", "invalid-state01"): 1.0
#             }
#         }
#         actions = {"some-action"}
#         states = {"state01"}

#         with self.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
#             FactoredTransitionFunction(transition_function=transition_function_with_invalid_to_state, actions=actions, state_variables=state_variables)

#         with self.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
#             FactoredTransitionFunction(transition_function=transition_function_with_invalid_to_state, actions=actions, state_variables=state_variables)

#     def test_constructor_call_with_transition_function_having_action_with_invalid_transition_distributions(self):
#         transition_function_with_transition_greater_than_one = {
#             "some-action": {
#                 ("state01", "state01"): 0.9,
#                 ("state01", "state02"): 0.2,
#                 ("state02", "state02"): 1.0
#             }
#         }
#         transition_function_with_transition_lesser_than_one = {
#             "some-action": {
#                 ("state01", "state01"): 1.0,
#                 ("state02", "state02"): 0.1,
#             }
#         }
#         actions = {"some-action"}
#         states = {"state01","state02"}

#         with self.assertRaisesRegex(ValueError, "Invalid probability distribution on \[state01\] transition in action \[some-action\]. The sum of all transitions probabilities from this state to others must be 1 \(one\)"):
#             FactoredTransitionFunction(transition_function=transition_function_with_transition_greater_than_one, actions=actions, state_variables=state_variables)

#         with self.assertRaisesRegex(ValueError, "Invalid probability distribution on \[state02\] transition in action \[some-action\]. The sum of all transitions probabilities from this state to others must be 1 \(one\)"):
#             FactoredTransitionFunction(transition_function=transition_function_with_transition_lesser_than_one, actions=actions, state_variables=state_variables)

#     def test_constructor_call_with_transition_function_having_action_with_invalid_transition_distributions(self):
#         transition_function_as_dict = {
#             "some-action": {
#                 ("state01", "state01"): 0.9,
#                 ("state01", "state02"): 0.1,
#                 ("state02", "state02"): 1.0
#             }
#         }
#         actions = {"some-action"}
#         states = {"state01","state02"}

#         FactoredTransitionFunction(transition_function=transition_function_as_dict, actions=actions, state_variables=state_variables)

#     # get_transition_probability method tests

#     def test_get_transition_probability_call_with_invalid_state(self):
#         transition_function_as_dict = {
#             "some-action": {
#                 ("state01", "state01"): 0.9,
#                 ("state01", "state02"): 0.1,
#                 ("state02", "state02"): 1.0
#             }
#         }
#         actions = {"some-action"}
#         states = {"state01","state02"}

#         transition_function = FactoredTransitionFunction(transition_function=transition_function_as_dict, actions=actions, state_variables=state_variables)

#         with self.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
#             transition_function.get_transition_probability("invalid-state01", "some-action", "state01")

#         with self.assertRaisesRegex(ValueError, "State \[invalid-state01\] not found in state set"):
#             transition_function.get_transition_probability("state01", "some-action", "invalid-state01")

#     def test_get_transition_probability_call_with_invalid_action(self):
#         transition_function_as_dict = {
#             "some-action": {
#                 ("state01", "state01"): 0.9,
#                 ("state01", "state02"): 0.1,
#                 ("state02", "state02"): 1.0
#             }
#         }
#         actions = {"some-action"}
#         states = {"state01","state02"}

#         transition_function = FactoredTransitionFunction(transition_function=transition_function_as_dict, actions=actions, state_variables=state_variables)

#         with self.assertRaisesRegex(ValueError, "Action \[some\-invalid\-action\] not found"):
#             transition_function.get_transition_probability("state01", "some-invalid-action", "state01")

#     def test_get_transition_probability_call_with_valid_parameters(self):
#         transition_function_as_dict = {
#             "some-action": {
#                 ("state01", "state01"): 0.9,
#                 ("state01", "state02"): 0.1,
#                 ("state02", "state02"): 1.0
#             }
#         }
#         actions = {"some-action"}
#         states = {"state01","state02"}

#         transition_function = FactoredTransitionFunction(transition_function=transition_function_as_dict, actions=actions, state_variables=state_variables)

#         self.assertEqual(transition_function.get_transition_probability("state01", "some-action", "state01"), 0.9)
#         self.assertEqual(transition_function.get_transition_probability("state01", "some-action", "state02"), 0.1)
#         self.assertEqual(transition_function.get_transition_probability("state02", "some-action", "state02"), 1.0)

#     # get_transition_matrix method tests

#     def test_get_transition_matrix_call_with_invalid_action(self):
#         transition_function_as_dict = {
#             "some-action": {
#                 ("state01", "state01"): 0.9,
#                 ("state01", "state02"): 0.1,
#                 ("state02", "state02"): 1.0
#             }
#         }
#         actions = {"some-action"}
#         states = {"state01","state02"}

#         transition_function = FactoredTransitionFunction(transition_function=transition_function_as_dict, actions=actions, state_variables=state_variables)

#         with self.assertRaisesRegex(ValueError, "Action \[some\-invalid\-action\] not found"):
#             transition_function.get_transition_matrix("some-invalid-action")

#     def test_get_transition_matrix_call_with_valid_parameters(self):
#         transition_function_as_dict = {
#             "some-action": {
#                 ("state01", "state01"): 0.9,
#                 ("state01", "state02"): 0.1,
#                 ("state02", "state02"): 1.0
#             }
#         }
#         actions = {"some-action"}
#         states = {"state01","state02"}

#         transition_function = FactoredTransitionFunction(transition_function=transition_function_as_dict, actions=actions, state_variables=state_variables)

#         expected_matrix = np.matrix([[0.9, 0.1], [0.0, 1.0]])
#         obtained_matrix = transition_function.get_transition_matrix("some-action")

#         self.assertListEqual(list(obtained_matrix.flat), list(expected_matrix.flat))
