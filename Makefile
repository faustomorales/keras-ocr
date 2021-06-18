.PHONY: docs

PKG_NAME:=keras_ocr

# Select specific Python tests to run using pytest selectors
# e.g., make test TEST_SCOPE='-m "not_integration" tests/api/'
TEST_SCOPE?=tests/

# Prefix for running commands on the host vs in Docker (e.g., dev vs CI)
EXEC:=poetry run
SPHINX_AUTO_EXTRA:=


help:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z0-9_%/-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Tips"
	@echo "----"
	@echo '- Run `make shell` to activate the project virtualenv in your shell'
	@echo '  e.g., make test TEST_SCOPE="-m not_integration tests/api/"'

init:  ## Initialize the development environment.
	pip install poetry poetry-dynamic-versioning
	poetry install

format-check: ## Make black check source formatting
	@$(EXEC) black --diff --check .

format: ## Make black unabashedly format source code
	@$(EXEC) black .

package: ## Make a local build of the Python package, source dist and wheel
	@mkdir -p dist
	@$(EXEC) poetry build

test: ## Make pytest run tests
	@$(EXEC) pytest -vxrs $(TEST_SCOPE)

type-check: ## Make mypy check types
	@$(EXEC) mypy $(PKG_NAME) tests

lint-check: ## Make pylint lint the package
	@$(EXEC) pylint --jobs 0 $(PKG_NAME)

lab: ## Start a jupyter lab instance
	@$(EXEC) jupyter lab

shell:  ## Jump into poetry shell.
	poetry shell

docs: ## Make a local HTML doc server that updates on changes to from Sphinx source
	@$(EXEC) pip install -r docs/requirements.txt
	@$(EXEC) sphinx-autobuild -b html docs docs/build/html $(SPHINX_AUTO_EXTRA)
