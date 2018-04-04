.PHONY: install docs


venv: venv/bin/activate
IN_VENV=. ./venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv venv --python=python3
	${IN_VENV} && pip install pip --upgrade
	${IN_VENV} && pip install numpy
	${IN_VENV} && pip install -r requirements.txt

install: venv
	${IN_VENV} && python setup.py install

