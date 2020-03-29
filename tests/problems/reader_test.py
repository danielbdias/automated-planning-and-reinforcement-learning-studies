from ..context import probabilistic_planning
from probabilistic_planning.problems.reader import read_problem_file

import unittest
from unittest import mock

from nose2.tools import such

def mock_file_content(lines):
    file_content = "\n".join(lines)
    mock_open = mock.mock_open(read_data=file_content)
    return mock.patch('probabilistic_planning.problems.reader.open', mock_open)


with such.A('Problem Reader function') as it:
    @it.should('fail when the problem file is empty')
    def test_empty_file(self):
        file_content = [
            ""
        ]

        with mock_file_content(file_content), \
             self.assertRaises(ValueError):
            read_problem_file('some_file.txt')

    @it.should('fail when the problem file is empty again')
    def test_empty_file_again(self):
        file_content = [
            ""
        ]

        with mock_file_content(file_content), \
             self.assertRaises(ValueError):
            read_problem_file('some_file.txt')

    it.createTests(globals())

if __name__ == '__main__':
    unittest.main()