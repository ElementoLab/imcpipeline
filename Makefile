.DEFAULT_GOAL := all

NAME=$(shell basename `pwd`)

help:  ## Display help and quit
	@echo Makefile for the $(NAME) package.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'

all: install clean test  ## Install and test package

clean_build:
	rm -rf build/

clean_dist:
	rm -rf dist/

clean_eggs:
	rm -rf *.egg-info

clean_docs:
	rm -rf docs/build/

clean: clean_dist clean_eggs clean_build clean_docs  ## remove built files

_install:
	python setup.py sdist
	python -m pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	python -m pip install dist/*-py3-none-any.whl --user --upgrade

install:  ## Cleanly install package
	${MAKE} clean
	${MAKE} _install
	${MAKE} clean

install_f:
	pip install . --upgrade --no-deps --force-reinstall

test:  ## Test imcpipeline by running it on demo data
	imcpipeline --demo


build:
	python setup.py sdist bdist_wheel

pypitest: build
	twine \
		upload \
		-r pypitest dist/*

pypi: build  ## Upload to PyPI
	twine \
		upload \
		dist/*


sync:  ## Sync to to cluster
	rsync --copy-links --progress -r \
	. afr4001@pascal.med.cornell.edu:projects/imcpipeline

.PHONY : clean_build clean_dist clean_eggs clean_docs clean _install install install_f test build pypitest pypi sync
