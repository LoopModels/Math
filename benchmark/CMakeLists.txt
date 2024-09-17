cmake_minimum_required(VERSION 3.23)

project(MathBenchmarks LANGUAGES C CXX)

option(ENABLE_NATIVE_COMPILATION "Compile with -march=native" ON)
option(ENABLE_WIDE_VECTORS "Compile with 512bit vectors if available" ON)
option(ENABLE_OPENMP "Use OpenMP for a multithreading benchmark" OFF)
option(POLYMATHNOEXPLICITSIMDARRAY "No explicit SIMD for Array operations" OFF)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

# ---- compile_commands.json ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cpmaddpackage(
  NAME
  benchmark
  GITHUB_REPOSITORY
  google/benchmark
  VERSION
  1.8.4
  OPTIONS
  "BENCHMARK_ENABLE_TESTING Off"
  "BENCHMARK_ENABLE_LIBPFM On"
  "BENCHMARK_ENABLE_WERROR Off"
  "BENCHMARK_ENABLE_EXCEPTIONS Off"
  SYSTEM
  TRUE)
if(benchmark_ADDED)
  # enable c++11 to avoid compilation errors
  set_target_properties(benchmark PROPERTIES CXX_STANDARD 11)
endif()

# FetchContent_Declare( Math GIT_REPOSITORY git@github.com:LoopModels/Math.git
# GIT_TAG origin/main ) FetchContent_MakeAvailable(Math)

# file(GLOB_RECURSE headers CONFIGURE_DEPENDS
# ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
file(GLOB benchmarks CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${benchmarks})

message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
add_subdirectory("${PROJECT_SOURCE_DIR}/.." "extern_build/math")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# --- Maybe Use SIMD Array Ops ---
if(POLYMATHNOEXPLICITSIMDARRAY)
  target_compile_definitions(${PROJECT_NAME} PUBLIC POLYMATHNOEXPLICITSIMDARRAY)
  message(STATUS "Do not use SIMD Array operations")
else()
  message(STATUS "Use SIMD Array operations")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE Math benchmark::benchmark_main)

if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES
                                               "IntelLLVM"))
  target_compile_options(${PROJECT_NAME} PRIVATE -ferror-limit=2
                                                 -fcolor-diagnostics)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(
    ${PROJECT_NAME}
    PRIVATE -fmax-errors=2 -fconcepts-diagnostics-depth=4
            -fno-semantic-interposition -fdiagnostics-color=always
            -fverbose-asm)
endif()

if(ENABLE_NATIVE_COMPILATION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(${PROJECT_NAME} PRIVATE -xhost)
    if(ENABLE_WIDE_VECTORS)
      target_compile_options(${PROJECT_NAME} PRIVATE -qopt-zmm-usage=high)
    endif()
  else()
    target_compile_options(${PROJECT_NAME} PRIVATE -march=native
    )# -fno-unroll-loops)
    if(ENABLE_WIDE_VECTORS)
      check_cxx_compiler_flag("-mprefer-vector-width=512" VEC512)
      if(VEC512)
        target_compile_options(${PROJECT_NAME}
                               PRIVATE -mprefer-vector-width=512)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
          if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${PROJECT_NAME}
                                   PRIVATE -mtune-ctrl=avx512_move_by_pieces)
          endif()
        endif()
      endif()
    endif()
  endif()
endif()
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 23
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(
  ${PROJECT_NAME} PROPERTIES ENVIRONMENT
                             WORKING_DIRECTORY=${PROJECT_BINARY_DIR})

target_compile_options(
  ${PROJECT_NAME}
  PRIVATE -fno-exceptions
          -fno-rtti
          -fstrict-aliasing
          -fno-plt
          -fstrict-overflow
          -fomit-frame-pointer
          -fno-signed-zeros
          -fassociative-math
          -ffinite-math-only
          -funsafe-math-optimizations
          -fno-trapping-math
          -Wall
          -Wshadow
          -Wextra
          -save-temps
          -Werror)
if(ENABLE_OPENMP)
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(${PROJECT_NAME} PRIVATE -fiopenmp)
  else()
    target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
  endif()
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp-simd)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  target_compile_options(${PROJECT_NAME} PRIVATE -masm=intel)
endif()
