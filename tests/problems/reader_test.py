from ..context import probabilistic_planning
from probabilistic_planning.problems.reader import read_problem_file

import unittest
from unittest import mock

from nose2.tools import such

def mock_file_content(lines):
    file_content = "\n".join(lines)
    mock_open = mock.mock_open(read_data=file_content)
    return mock.patch("probabilistic_planning.problems.reader.open", mock_open)


with such.A("Problem Reader function") as it:

    @it.should("fail when the problem file does not define any sections")
    def test_file_with_any_section(self):
        file_content = [
            ""
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(ValueError, "The states should be defined"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define state section")
    def test_file_without_state_section(self):
        file_content = [
            ""
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(ValueError, "The states should be defined"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define state section end")
    def test_file_without_state_section_end(self):
        file_content = [
            "states",
            "   state01"
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(Exception, "endstates token not found"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define reward and action sections")
    def test_file_without_reward_and_action_sections(self):
        file_content = [
            "states",
            "   state01",
            "endstates"
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(ValueError, "The reward function should be defined"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define reward section end")
    def test_file_without_reward_section_end(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0"
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(Exception, "endreward token not found"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define an action section")
    def test_file_without_action_section(self):
        file_content = [
            "states",
            "   state01",
            "endstates",
            "reward",
            "\tstate01 1.0",
            "endreward"
        ]

        with mock_file_content(file_content), \
             it.assertRaisesRegex(ValueError, "The transition function must have at least one action transition matrix"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define action section end")
    def test_file_without_action_section_end(self):
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
             it.assertRaisesRegex(Exception, "endaction token not found"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define initial state section end")
    def test_file_without_initial_state_section_end(self):
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
             it.assertRaisesRegex(Exception, "endinitialstate token not found"):
            read_problem_file("some_file.txt")

    @it.should("fail when the problem file does not define goal state section end")
    def test_file_without_goal_state_section_end(self):
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
             it.assertRaisesRegex(Exception, "endgoalstate token not found"):
            read_problem_file("some_file.txt")

    @it.should("returns a parsed mdp when the problem file has all required sections")
    def test_file_with_required_sections(self):
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
            mdp = read_problem_file("some_file.txt")

            it.assertSetEqual(mdp.states, {"state01"})
            it.assertDictEqual(mdp.reward_function, { "state01": 1.0 })

            it.assertEqual(mdp.transition_function.get_transition_probability("state01", "first-action", "state01"), 1.0)

    @it.should("returns a parsed MDP when the problem file has all sections")
    def test_file_with_all_sections(self):
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
            mdp = read_problem_file("some_file.txt")

            it.assertSetEqual(mdp.states, {"state01"})
            it.assertDictEqual(mdp.reward_function, { "state01": 1.0 })

            it.assertEqual(mdp.transition_function.get_transition_probability("state01", "first-action", "state01"), 1.0)

            it.assertSetEqual(mdp.initial_states, {"state01"})
            it.assertSetEqual(mdp.goal_states, {"state01"})

    it.createTests(globals())
