# Matt

### Build
```shell
make build
```

### Format
```shell
make format
```

### Testing
```shell
make test

# specific test
./build/tests --gtest_list_tests
./build/tests --gtest_filter="OpsTest.Matmul*"
```