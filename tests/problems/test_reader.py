from ..context import probabilistic_planning
from probabilistic_planning.problems import reader

import unittest
from unittest import mock

def mock_file_content(lines):
    file_content = "\n".join(lines)
    mock_open = mock.mock_open(read_data=file_content)
    return mock.patch("probabilistic_planning.problems.reader.open", mock_open)

class ProblemReaderTests(unittest.TestCase):

    def test_read_problem_file_call_with_file_with_any_section(self):
        file_content = [
            ""
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(ValueError, "The states should be defined"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_without_state_section(self):
        file_content = [
            ""
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(ValueError, "The states should be defined"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_without_state_section_end(self):
        file_content = [
            "states",
            "   state01"
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(Exception, "endstates token not found"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_without_reward_and_action_sections(self):
        file_content = [
            "states",
            "   state01",
            "endstates"
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(ValueError, "The reward function should be defined"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_without_reward_section_end(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0"
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(Exception, "endreward token not found"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_without_action_section(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0",
            "endreward"
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(ValueError, "The transition function must have at least one action transition matrix"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_without_action_section_end(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0",
            "endreward",
            "action first-action",
            "\tstate01 state01 1.0"
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(Exception, "endaction token not found"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_without_initial_state_section_end(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0",
            "endreward",
            "action first-action",
            "\tstate01 state01 1.0",
            "endaction",
            "initialstate",
            "   state01"
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(Exception, "endinitialstate token not found"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_without_goal_state_section_end(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0",
            "endreward",
            "action first-action",
            "\tstate01 state01 1.0",
            "endaction",
            "initialstate",
            "   state01",
            "endinitialstate",
            "goalstate",
            "   state01"
        ]

        with mock_file_content(file_content), \
             self.assertRaisesRegex(Exception, "endgoalstate token not found"):
            reader.read_problem_file("some_file.txt")

    def test_read_problem_file_call_with_file_with_required_sections(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0",
            "endreward",
            "action first-action",
            "\tstate01 state01 1.0",
            "endaction"
        ]

        with mock_file_content(file_content):
            mdp = reader.read_problem_file("some_file.txt")

            self.assertSetEqual(mdp.states, {"state01"})
            self.assertDictEqual(mdp.reward_function, { "state01": 1.0 })

            self.assertEqual(mdp.transition_function.get_transition_probability("state01", "first-action", "state01"), 1.0)

    def test_read_problem_file_call_with_file_with_all_sections(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "",
            "reward",
            "\tstate01 1.0",
            "endreward",
            "",
            "action first-action",
            "\tstate01 state01 1.0",
            "endaction",
            "",
            "initialstate",
            "   state01",
            "endinitialstate",
            "",
            "goalstate",
            "   state01",
            "endgoalstate"
        ]

        with mock_file_content(file_content):
            mdp = reader.read_problem_file("some_file.txt")

            self.assertSetEqual(mdp.states, {"state01"})
            self.assertDictEqual(mdp.reward_function, { "state01": 1.0 })

            self.assertEqual(mdp.transition_function.get_transition_probability("state01", "first-action", "state01"), 1.0)

            self.assertSetEqual(mdp.initial_states, {"state01"})
            self.assertSetEqual(mdp.goal_states, {"state01"})
