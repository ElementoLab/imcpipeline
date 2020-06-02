.DEFAULT_GOAL := all

all: install clean test

clean_build:
	rm -rf build/

clean_dist:
	rm -rf dist/

clean_eggs:
	rm -rf *.egg-info

clean: clean_dist clean_eggs clean_build

_install:
	python setup.py sdist
	python -m pip wheel --no-index --no-deps --wheel-dir dist dist/*.tar.gz
	python -m pip install dist/*-py3-none-any.whl --user --upgrade

install:
	${MAKE} clean
	${MAKE} _install
	${MAKE} clean

test:
	imcpipeline --demo


build:
	python setup.py sdist bdist_wheel

pypitest: build
	twine \
		upload \
		-r pypitest dist/*

pypi: build
	twine \
		upload \
		dist/*


.PHONY : clean_build clean_dist clean_eggs clean _install install test build pypitest pypi
