cmake_minimum_required(VERSION 3.16)
project(otrep_x_prime VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# C++ Standard and Compiler Options
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif()

# =============================================================================
# Dependencies
# =============================================================================

# Find Eigen3 (header-only)
find_package(Eigen3 3.4 QUIET)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching from repository...")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

# Find pybind11 for Python bindings
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from repository...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# =============================================================================
# Core Library
# =============================================================================

add_library(otrep_core_lib STATIC
    src/HybridStrategy.cpp
    src/MarketGraph.cpp
)

target_include_directories(otrep_core_lib 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(otrep_core_lib 
    PUBLIC 
        Eigen3::Eigen
)

# Enable SIMD if available
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(otrep_core_lib PRIVATE -mavx2 -mfma)
endif()

# =============================================================================
# Benchmark Executable
# =============================================================================

add_executable(benchmark_strategy
    benchmark/benchmark_main.cpp
)

target_link_libraries(benchmark_strategy PRIVATE otrep_core_lib)

# =============================================================================
# Test Executable (optional)
# =============================================================================

enable_testing()

add_executable(test_strategy
    test/test_main.cpp
)

target_link_libraries(test_strategy PRIVATE otrep_core_lib)

add_test(NAME StrategyTests COMMAND test_strategy)

# =============================================================================
# Python Bindings (pybind11)
# =============================================================================

pybind11_add_module(otrep_core
    python_bindings/otrep_wrapper.cpp
)

target_link_libraries(otrep_core 
    PRIVATE 
        otrep_core_lib
        Eigen3::Eigen
)

# Enable SIMD for Python module
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(otrep_core PRIVATE -mavx2 -mfma)
endif()

# Set output directory for easy Python import
set_target_properties(otrep_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../"
)

# =============================================================================
# Install
# =============================================================================

install(TARGETS otrep_core_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/otrep
    DESTINATION include
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== OTREP-X PRIME C++ Build Configuration ===")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Eigen Found:    ${Eigen3_FOUND}")
message(STATUS "  pybind11 Found: ${pybind11_FOUND}")
message(STATUS "  Python Module:  otrep_core.*.so -> ${CMAKE_SOURCE_DIR}/../")
message(STATUS "")
