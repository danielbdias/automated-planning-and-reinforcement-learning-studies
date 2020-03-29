install.dependencies:
	pip install -r requirements.txt

run.tests:
	nose2 --verbose

run.tests.watch:
	watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command='clear && make run.tests' \
		.

run.lint:
	pylint --output-format=colorized probabilistic_planning

run.lint.watch:
	watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command='clear && make run.lint' \
		.
