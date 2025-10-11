cmake_minimum_required(VERSION 3.14)

project(cpp_geoguessr LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# Find required packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV development packages.")
endif()

# Find or download pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find ONNX Runtime
# First, try to find ONNX Runtime using standard paths
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
        /opt/homebrew/include/onnxruntime
        ${CMAKE_SOURCE_DIR}/../onnxruntime/include
    DOC "Path to ONNX Runtime include directory"
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        /usr/local/lib
        /usr/lib
        /opt/homebrew/lib
        ${CMAKE_SOURCE_DIR}/../onnxruntime/lib
    DOC "Path to ONNX Runtime library"
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(WARNING "ONNX Runtime not found. You'll need to install it manually.")
    message(STATUS "Expected locations:")
    message(STATUS "  Include: /usr/local/include/onnxruntime or /opt/homebrew/include/onnxruntime")
    message(STATUS "  Library: /usr/local/lib/libonnxruntime.so or /opt/homebrew/lib/libonnxruntime.dylib")
    message(STATUS "")
    message(STATUS "Download from: https://github.com/microsoft/onnxruntime/releases")
    message(STATUS "Or install via package manager:")
    message(STATUS "  macOS: brew install onnxruntime")
    message(STATUS "  Ubuntu: Download and extract to /usr/local/")
    
    # Set placeholder paths for now - user will need to install ONNX Runtime
    set(ONNXRUNTIME_INCLUDE_DIR "/usr/local/include/onnxruntime")
    set(ONNXRUNTIME_LIBRARY "/usr/local/lib/libonnxruntime.so")
else()
    message(STATUS "Found ONNX Runtime:")
    message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "  Library: ${ONNXRUNTIME_LIBRARY}")
endif()

# Define source files
set(SOURCES
    src/predictor.cpp
    src/bindings.cpp
)

set(HEADERS
    src/predictor.h
)

# Create the pybind11 module
pybind11_add_module(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIBRARY}
)

# Compiler-specific options
target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO="${PROJECT_VERSION}")

# Set properties for the Python module
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
)

# Platform-specific settings
if(APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# Create a setup script for easier building
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/build_setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/build_setup.py @ONLY)

# Print build information
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "===========================")