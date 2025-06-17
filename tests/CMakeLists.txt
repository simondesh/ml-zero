# Test executable sources
set(TEST_SOURCES
    test_main.cpp
    unit_tests/test_vector.cpp
    # unit_tests/test_matrix.cpp
    # unit_tests/test_statistics.cpp
    # Add more test files as you create them
)

# Create test executable
add_executable(mlzero_tests ${TEST_SOURCES})

# Link against our library and Catch2
target_link_libraries(mlzero_tests 
    PRIVATE 
    mlzero 
    catch2
)

# Include directories for tests
target_include_directories(mlzero_tests PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

# Register tests with CTest
include(CTest)
add_test(NAME all_tests COMMAND mlzero_tests)

# Individual test discovery (optional but recommended)
# include(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catch2/ParseAndAddCatchTests.cmake)
# if(COMMAND ParseAndAddCatchTests)
#     ParseAndAddCatchTests(mlzero_tests)
# endif()

# Specific test categories
add_test(NAME vector_tests COMMAND mlzero_tests "[vector]")
# add_test(NAME matrix_tests COMMAND mlzero_tests "[matrix]")
# add_test(NAME statistics_tests COMMAND mlzero_tests "[statistics]")

# # Performance benchmarks (separate executable)
# add_executable(mlzero_benchmarks 
#     benchmarks/benchmark_main.cpp
#     benchmarks/benchmark_algorithms.cpp
# )

# target_link_libraries(mlzero_benchmarks 
#     PRIVATE 
#     mlzero 
#     catch2
# )

# # Don't run benchmarks by default (they take time)
# add_test(NAME benchmarks COMMAND mlzero_benchmarks)
# set_tests_properties(benchmarks PROPERTIES LABELS "benchmark")