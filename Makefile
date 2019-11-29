PACKAGE_NAME = keras-ocr
IMAGE_NAME = $(PACKAGE_NAME)
VOLUME_NAME = $(IMAGE_NAME)_venv
DOCKER_ARGS = -v $(PWD):/usr/src -v $(VOLUME_NAME):/usr/src/.venv --rm
IN_DOCKER = docker run $(DOCKER_ARGS) $(IMAGE_NAME) pipenv run
NOTEBOOK_PORT ?= 5000
JUPYTER_OPTIONS := --ip=0.0.0.0 --port $(NOTEBOOK_PORT) --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
TEST_SCOPE ?= tests/
DOCUMENTATION_PORT ?= 5001

.PHONY: build
build:
	docker build --rm --force-rm -t $(IMAGE_NAME) .
	@-docker volume rm $(VOLUME_NAME)
init:
	# Blow away the venv to deal with pip caching issues with conflicting
	# versions of OpenCV.
	PIPENV_VENV_IN_PROJECT=true pipenv install --dev --skip-lock
	pipenv run pip uninstall opencv-python -y
	pipenv run pip install -U  --no-cache-dir opencv-contrib-python-headless
	pipenv run pip install -r docs/requirements.txt
bash:
	docker run -it $(DOCKER_ARGS) $(IMAGE_NAME) bash
command-lab-server:
	pipenv run jupyter lab $(JUPYTER_OPTIONS)
lab-server:
	docker run -it $(DOCKER_ARGS) -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) $(IMAGE_NAME) make command-lab-server NOTEBOOK_PORT=$(NOTEBOOK_PORT)
documentation-server:
	docker run -it $(DOCKER_ARGS) -p $(DOCUMENTATION_PORT):$(DOCUMENTATION_PORT) $(IMAGE_NAME) pipenv run sphinx-autobuild -b html "docs" "docs/_build/html" --host 0.0.0.0 --port $(DOCUMENTATION_PORT) $(O)
test:
	$(IN_DOCKER) pytest $(TEST_SCOPE)
format:
	$(IN_DOCKER) yapf --recursive --in-place --exclude=keras_ocr/_version.py tests keras_ocr
lint_check:
	$(IN_DOCKER) pylint -j 0 keras_ocr --rcfile=setup.cfg
format_check:
	$(IN_DOCKER) yapf --recursive --exclude=keras_ocr/_version.py --diff tests keras_ocr || (echo '\nUnexpected format.' && exit 1)
precommit:
	make build
	make lint_check
	make format_check
	make test