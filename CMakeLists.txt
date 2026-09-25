cmake_minimum_required(VERSION 3.16)
project(edgelm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(EDGELM_SANITIZE "Build with AddressSanitizer + UBSan" OFF)
option(EDGELM_NATIVE "Tune for the build machine (-mcpu=native)" OFF)
option(EDGELM_TESTS "Build the doctest suite" ON)

find_package(Threads REQUIRED)

add_library(edgelm STATIC
  src/graph.cpp
  src/model.cpp
  src/passes.cpp
  src/thread_pool.cpp
  src/kernels_ref.cpp
  src/kernels_cpu.cpp
  src/backends.cpp
  src/profiler.cpp
  src/runtime.cpp
  src/tokenizer.cpp)
target_include_directories(edgelm PUBLIC include)
target_compile_options(edgelm PUBLIC -Wall -Wextra -Wpedantic)
target_link_libraries(edgelm PUBLIC Threads::Threads)
if(EDGELM_NATIVE)
  target_compile_options(edgelm PUBLIC -mcpu=native)
endif()
if(EDGELM_SANITIZE)
  target_compile_options(edgelm PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(edgelm PUBLIC -fsanitize=address,undefined)
endif()

add_executable(edgelm_cli src/main.cpp)
set_target_properties(edgelm_cli PROPERTIES OUTPUT_NAME edgelm)
target_link_libraries(edgelm_cli PRIVATE edgelm)

if(EDGELM_TESTS)
  enable_testing()
  file(GLOB EDGELM_TEST_SOURCES tests/test_*.cpp)
  add_executable(edgelm_tests ${EDGELM_TEST_SOURCES})
  target_link_libraries(edgelm_tests PRIVATE edgelm)
  add_test(NAME edgelm_tests COMMAND edgelm_tests)
  set_tests_properties(edgelm_tests PROPERTIES ENVIRONMENT "EDGELM_TESTDATA=${CMAKE_SOURCE_DIR}/testdata")
endif()
