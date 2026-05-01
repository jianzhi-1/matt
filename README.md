# Matt
```shell
cmake -S . -B build
cmake --build build
./build/test
```

### Format
```shell
cmake --build build --target format
```

### Testing
```shell
cmake -S . -B build
cmake --build build
./build/tests

# specific test
./build/tests --gtest_list_tests
./build/tests --gtest_filter="OpsTest.Matmul*"
```