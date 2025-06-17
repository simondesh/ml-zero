# catch2 library setup
add_library(catch2 catch_amalgamated.cpp)
target_include_directories(catch2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# create interface 
add_library(catch2_interface INTERFACE)
target_include_directories(catch2_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})