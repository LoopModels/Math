cmake_minimum_required(VERSION 3.28.2)

# # ---- Static Analyzers ---- find_program(CLANGTIDY clang-tidy) if(NOT
# CLANGTIDY MATCHES CLANGTIDY-NOTFOUND) set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY})
# endif() find_program(CPPCHECK cppcheck) if(NOT CPPCHECK MATCHES
# CPPCHECK-NOTFOUND) set(CMAKE_CXX_CPPCHECK "${CPPCHECK} --language=c++
# --enable=warning,style,performance,information") endif()

project(MathTests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)
option(POLYMATHNOEXPLICITSIMDARRAY "No explicit SIMD for Array operations" OFF)
option(ENABLE_LLD "Use lld as the linker" ON)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

# ---- compile_commands.json ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(((USE_SANITIZER MATCHES "([Aa]ddress)") OR (USE_SANITIZER MATCHES
                                               "([Aa]ddress);([Uu]ndefined)"))
   AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  find_library(LUNWIND_LIB unwind)
  message(STATUS "Found unwindlib ${LUNWIND_LIB}")
  if(LUNWIND_LIB)
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -lunwind -Wno-unused-command-line-argument")
  endif()
endif()

include(FetchContent)
# using `FIND_PACKAGE_ARGS` causes it to try `find_package`
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 305e5a238b3c8d11266fbafd85520fb6b3184851 # May 24, 2024 main
  FIND_PACKAGE_ARGS NAMES GTest)
FetchContent_MakeAvailable(googletest)

cpmaddpackage("gh:TheLartians/Format.cmake@1.7.3")

if((USE_SANITIZER MATCHES "([Uu]ndefined)") OR (USE_SANITIZER MATCHES
                                                "([Aa]ddress);([Uu]ndefined)"))
  set(ENABLE_RTTI
      ON
      CACHE BOOL "Use RTTI")
  set(USE_MI_MALLOC
      OFF
      CACHE BOOL "No mimalloc with sanitizers")
  set(USE_JE_MALLOC
      OFF
      CACHE BOOL "No jemalloc with sanitizers")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(ENABLE_STACK_PROTECTOR
        ON
        CACHE BOOL "Use stack potection")
  endif()
else()
  set(ENABLE_STACK_PROTECTOR
      ON
      CACHE BOOL "Use stack potection")
endif()
set(FOR_TESTING
    ON
    CACHE BOOL "Build for testing")
add_subdirectory("${PROJECT_SOURCE_DIR}/.." "extern_build/math")

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} Math GTest::gtest_main)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 23)

target_compile_options(${PROJECT_NAME} PRIVATE -fno-exceptions)
if((USE_SANITIZER MATCHES "([Uu]ndefined)") OR (USE_SANITIZER MATCHES
                                                "([Aa]ddress);([Uu]ndefined)"))
  target_compile_options(${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer
                                                 -fsanitize-trap=all)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -fno-rtti)
endif()
if(POLYMATHNOEXPLICITSIMDARRAY)
  target_compile_definitions(${PROJECT_NAME}
                             PRIVATE POLYMATHNOEXPLICITSIMDARRAY)
endif()

# enable compiler warnings
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES
                                              "GNU")
    # target_compile_options(Math PUBLIC -Wall -Wpedantic -Wextra -Wshadow
    # -Werror)
    target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wpedantic -Wextra
                                                  -Wshadow -Werror)
  elseif(MSVC)
    # target_compile_options(Math PUBLIC /W4 /WX)
    target_compile_definitions(${PROJECT_NAME}
                               PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
  endif()
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(
    ${PROJECT_NAME} PRIVATE -ferror-limit=8 -fcolor-diagnostics
                            -fstack-protector-strong -ftime-trace)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(
    ${PROJECT_NAME}
    PRIVATE -fmax-errors=8 -fconcepts-diagnostics-depth=4
            -fno-semantic-interposition -fdiagnostics-color=always
            -Wno-comma-subscript -Wno-psabi)
  if(NOT ((USE_SANITIZER MATCHES "([Aa]ddress)")
          OR (USE_SANITIZER MATCHES "([Aa]ddress);([Uu]ndefined)")))
    target_compile_options(${PROJECT_NAME} PRIVATE -fstack-protector-strong)
  endif()
endif()
# target_compile_options(Math INTERFACE -D_GLIBCXX_ASSERTIONS)
target_compile_options(${PROJECT_NAME} PRIVATE -D_GLIBCXX_ASSERTIONS
                                               -ftemplate-backtrace-limit=0)
if(ENABLE_LLD)
  # TODO: use `CMAKE_LINKER_TYPE`/`LINKER_TYPE` once you've upgraded to cmake 29
  target_link_options(${PROJECT_NAME} PRIVATE -fuse-ld=lld)
endif()

# ---- Add MathTests ----

enable_testing()

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})

# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  # target_compile_options(Math PUBLIC -O0 -g --coverage)
  # target_link_options(Math PUBLIC --coverage)
  target_compile_options(${PROJECT_NAME} PUBLIC -O0 -g --coverage)
  target_link_options(${PROJECT_NAME} PUBLIC --coverage)
  add_custom_target(
    clean_coverage
    COMMAND rm -f ${PROJECT_BINARY_DIR}/CMakeFiles/MathTests.dir/*.gcda
    COMMAND
      rm -f
      ${PROJECT_BINARY_DIR}/_deps/math-build/CMakeFiles/Math.dir/lib/*.gcda
    COMMENT "Clearing out old coverage data.")
  add_dependencies(${PROJECT_NAME} clean_coverage)
  message(STATUS "Test coverage enabled.")
endif()
