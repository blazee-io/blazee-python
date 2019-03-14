.PHONY: build

clean:
	rm -rf build
	rm -rf dist

build:
	$(MAKE) clean
	pipenv run python3 setup.py sdist bdist_wheel

dist_pypi:
	$(MAKE) build
	pipenv run twine upload dist/*

test:
	env $(cat .env | xargs) pipenv run pytest -s tests
