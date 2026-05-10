.PHONY: build test pytest format clean

build:
	cmake -S . -B build
	cmake --build build

test: build
	./build/tests

pytest: build
	PYTHONPATH=build python -m pytest tests/test_matt.py -v

format:
	cmake --build build --target format

clean:
	rm -rf build