#!/bin/bash
set -e

echo "Building MlZero ..."

# create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build the project
echo "Building..."
make -j$(nproc)

echo "Build completed successfully!"

# Run tests if requested
if [ "$1" = "test" ]; then
    echo "Running tests..."
    make test
    echo "All tests passed!"
fi

# Run example if requested
if [ "$1" = "example" ]; then
    echo "Running vector example..."
    ./examples/basic_usage/vector_example
fi

echo "Done!"