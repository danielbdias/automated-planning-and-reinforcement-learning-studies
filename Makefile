install.dependencies:
	pip install -r requirements.txt

run.tests:
	nose2 --start-dir ./tests --project-directory ./probabilistic_planning --coverage ./probabilistic_planning

run.tests.coverage:
	nose2 --start-dir ./tests --project-directory ./probabilistic_planning --coverage ./probabilistic_planning --coverage-report html --coverage-report term --with-coverage

run.tests.watch:
	watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command='clear && make run.tests' \
		.

run.lint:
	pylint --rcfile=.pylint.cfg probabilistic_planning

run.lint.watch:
	watchmedo shell-command \
		--patterns="*.py" \
		--recursive \
		--command='clear && make run.lint' \
		.

build.docs:
	sphinx-build -b html probabilistic_planning docs
