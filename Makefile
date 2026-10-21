PY ?= python

.PHONY: test smoke reproduce data

test:
	$(PY) -m experiments.pipeline test

smoke:
	$(PY) -m experiments.pipeline smoke

reproduce:
	$(PY) -m experiments.pipeline reproduce

data:
	$(PY) -m experiments.pipeline data
