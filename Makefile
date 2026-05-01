.PHONY: build test format clean

build:
	cmake -S . -B build
	cmake --build build

test: build
	./build/tests

format:
	cmake --build build --target format

clean:
	rm -rf build