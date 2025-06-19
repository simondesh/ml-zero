
# Vector Example
add_executable(vector_example basic_usage/vector_example.cpp)
target_link_libraries(vector_example mlzero)
target_include_directories(vector_example PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Matrix Example
add_executable(matrix_example basic_usage/matrix_example.cpp)
target_link_libraries(matrix_example mlzero)
target_include_directories(matrix_example PRIVATE ${PROJECT_SOURCE_DIR}/include)

# TODO : Make linear algebra example