cmake_minimum_required(VERSION 3.14)

option(ENABLE_LLD "Use LLD? Off by default." OFF)
option(USE_MI_MALLOC "Use MI_MALLOC if available." ON)
option(USE_JE_MALLOC "Use jemallc if available and not using mi_malloc." ON)
# ---- Project ----

# Note: update this to your new project's name and version
project(
  Math
  VERSION 1.0
  LANGUAGES CXX
)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info

include(cmake/CPM.cmake)

# PackageProject.cmake will be used to make our target installable
CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.8.0")

# ---- Add source files ----

# Note: globbing sources is considered bad practice as CMake's generators may not detect new files
# automatically. Keep that in mind when changing files, or explicitly mention them here.
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

# ---- Create library ----

# Note: for header-only libraries change all PUBLIC flags to INTERFACE and create an interface
# target:
add_library(${PROJECT_NAME} INTERFACE ${headers})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 23)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

# --- Allocator override ----

set(SEARCH_JE_MALLOC USE_JE_MALLOC)
if(USE_MI_MALLOC)
  find_package(mimalloc 2.1 REQUIRED)
  if(mimalloc_FOUND)
    target_link_libraries(${PROJECT_NAME} INTERFACE mimalloc) # mimalloc-static
    target_compile_definitions(${PROJECT_NAME} INTERFACE USING_MIMALLOC)
    set(SEARCH_JE_MALLOC OFF)
  endif()
endif()
if(SEARCH_JE_MALLOC)
  find_package(PkgConfig)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(JEMALLOC jemalloc)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${JEMALLOC_LIBRARIES})
    target_include_directories(${PROJECT_NAME} INTERFACE ${JEMALLOC_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} INTERFACE USING_JEMALLOC)
  endif()
endif()
# ---- Create an installable target ----
# this allows users to install and find the library via `find_package()`.

# the location where the project's version header will be placed should match the project's regular
# header paths
string(TOLOWER ${PROJECT_NAME}/version.h VERSION_HEADER_LOCATION)

packageProject(
  NAME ${PROJECT_NAME}
  VERSION ${PROJECT_VERSION}
  NAMESPACE ${PROJECT_NAME}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include
  INCLUDE_DESTINATION include/${PROJECT_NAME}-${PROJECT_VERSION}
  VERSION_HEADER "${VERSION_HEADER_LOCATION}"
  COMPATIBILITY SameMajorVersion
)
