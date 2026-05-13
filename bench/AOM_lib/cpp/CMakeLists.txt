cmake_minimum_required(VERSION 3.16)
project(aompls CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Tighten warnings on GCC / Clang.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# aompls header-only library target.
add_library(aompls INTERFACE)
target_include_directories(aompls INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Disable Eigen's runtime asserts in Release for performance.
target_compile_definitions(aompls INTERFACE
    $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
    EIGEN_DONT_PARALLELIZE
)

option(AOMPLS_BUILD_TESTS "Build aompls C++ unit tests" ON)
option(AOMPLS_BUILD_EXAMPLES "Build aompls C++ examples" OFF)

if(AOMPLS_BUILD_TESTS)
    enable_testing()
    add_executable(test_operators tests/test_operators.cpp)
    target_link_libraries(test_operators PRIVATE aompls)
    add_test(NAME test_operators COMMAND test_operators)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_simpls.cpp)
        add_executable(test_simpls tests/test_simpls.cpp)
        target_link_libraries(test_simpls PRIVATE aompls)
        add_test(NAME test_simpls COMMAND test_simpls)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_selection.cpp)
        add_executable(test_selection tests/test_selection.cpp)
        target_link_libraries(test_selection PRIVATE aompls)
        add_test(NAME test_selection COMMAND test_selection)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_preproc.cpp)
        add_executable(test_preproc tests/test_preproc.cpp)
        target_link_libraries(test_preproc PRIVATE aompls)
        add_test(NAME test_preproc COMMAND test_preproc)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parity_kfold.cpp)
        add_executable(test_parity_kfold tests/test_parity_kfold.cpp)
        target_link_libraries(test_parity_kfold PRIVATE aompls)
        add_test(NAME test_parity_kfold COMMAND test_parity_kfold)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parity_spxy.cpp)
        add_executable(test_parity_spxy tests/test_parity_spxy.cpp)
        target_link_libraries(test_parity_spxy PRIVATE aompls)
        add_test(NAME test_parity_spxy COMMAND test_parity_spxy)
    endif()
endif()

if(AOMPLS_BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/basic.cpp)
    add_executable(aompls_basic examples/basic.cpp)
    target_link_libraries(aompls_basic PRIVATE aompls)
endif()
