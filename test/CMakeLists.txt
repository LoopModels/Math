cmake_minimum_required(VERSION 3.14)

project(MathTests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" ON)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

# ---- compile_commands.json ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.13.0
  VERSION 1.13.0 SYSTEM TRUE
  OPTIONS "INSTALL_GTEST OFF" "BUILD_GMOCK OFF" "gtest_hide_internal_symbols ON"
)
CPMAddPackage("gh:TheLartians/Format.cmake@1.7.3")

if(TEST_INSTALLED_VERSION)
  find_package(Math REQUIRED)
else()
  CPMAddPackage(NAME Math SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} Math::Math GTest::gtest_main)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

target_compile_options(${PROJECT_NAME} PRIVATE -fno-exceptions)
if((USE_SANITIZER MATCHES "([Uu]ndefined)") OR (USE_SANITIZER MATCHES "([Aa]ddress);([Uu]ndefined)"
                                               )
)
  # we use don't rtti, which is incompatible with the vptr check:
  # https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    target_compile_options(${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer -fsanitize-trap=all)
  else()
    target_compile_options(
      ${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer -fno-sanitize-recover=all
    )
  endif()
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -fno-rtti)
endif()

# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # target_compile_options(Math PUBLIC -Wall -Wpedantic -Wextra -Wshadow -Werror)
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wpedantic -Wextra -Wshadow -Werror)
  elseif(MSVC)
    # target_compile_options(Math PUBLIC /W4 /WX)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
  endif()
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(${PROJECT_NAME} PRIVATE -ferror-limit=8 -fcolor-diagnostics)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(
    ${PROJECT_NAME} PRIVATE -fmax-errors=8 -fconcepts-diagnostics-depth=4
                            -fno-semantic-interposition -fdiagnostics-color=always
  )
endif()
# target_compile_options(Math INTERFACE -D_GLIBCXX_ASSERTIONS)
target_compile_options(${PROJECT_NAME} PRIVATE -D_GLIBCXX_ASSERTIONS)
if(ENABLE_LLD)
  target_link_options(${PROJECT_NAME} PRIVATE -fuse-ld=lld)
  # target_link_options(Math PRIVATE -fuse-ld=lld)
endif()

target_precompile_headers(
  ${PROJECT_NAME}
  PRIVATE
  <algorithm>
  <cassert>
  <cmath>
  <cstddef>
  <cstdint>
  <limits>
  <ranges>
  <utility>
  <concepts>
  <gtest/gtest.h>
  <iterator>
  <limits>
  <numeric>
  <optional>
  <ostream>
  <random>
  <string>
  <tuple>
  <type_traits>
  <utility>
)

# ---- Add MathTests ----

enable_testing()

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})

# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  # target_compile_options(Math PUBLIC -O0 -g --coverage) target_link_options(Math PUBLIC
  # --coverage)
  target_compile_options(${PROJECT_NAME} PUBLIC -O0 -g --coverage)
  target_link_options(${PROJECT_NAME} PUBLIC --coverage)
  add_custom_target(
    clean_coverage
    COMMAND rm -f ${PROJECT_BINARY_DIR}/CMakeFiles/MathTests.dir/*.gcda
    COMMAND rm -f ${PROJECT_BINARY_DIR}/_deps/math-build/CMakeFiles/Math.dir/lib/*.gcda
    COMMENT "Clearing out old coverage data."
  )
  add_dependencies(${PROJECT_NAME} clean_coverage)
  message(STATUS "Test coverage enabled.")
endif()
