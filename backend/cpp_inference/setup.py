#!/usr/bin/env python3
"""
Setup script for the High-Performance C++ GeoGuessr Inference Engine.

This script provides an alternative to the bash build script and integrates
with pip for easier installation and distribution.
"""

import os
import sys
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext, Pybind11CMakeHelper
from setuptools import setup, Extension
import pybind11

# Project information
__version__ = "1.0.0"

project_dir = Path(__file__).parent.resolve()

# Check for required libraries
def check_opencv():
    """Check if OpenCV is available."""
    try:
        import cv2
        print(f"✅ OpenCV found: {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV not found. Install with: pip install opencv-python")
        return False

def find_onnxruntime():
    """Find ONNX Runtime installation."""
    possible_paths = [
        "/usr/local",
        "/opt/homebrew",
        "/usr"
    ]
    
    for base_path in possible_paths:
        include_path = Path(base_path) / "include" / "onnxruntime"
        lib_path = Path(base_path) / "lib"
        
        if include_path.exists():
            # Check for library files
            lib_files = list(lib_path.glob("libonnxruntime.*"))
            if lib_files:
                print(f"✅ ONNX Runtime found in {base_path}")
                return str(include_path), str(lib_path)
    
    print("❌ ONNX Runtime not found")
    print("Install with: brew install onnxruntime (macOS)")
    print("Or download from: https://github.com/microsoft/onnxruntime/releases")
    return None, None

def get_opencv_flags():
    """Get OpenCV compile and link flags."""
    try:
        # Try pkg-config first
        result = subprocess.run(['pkg-config', '--cflags', '--libs', 'opencv4'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            flags = result.stdout.strip().split()
            include_dirs = [flag[2:] for flag in flags if flag.startswith('-I')]
            library_dirs = [flag[2:] for flag in flags if flag.startswith('-L')]
            libraries = [flag[2:] for flag in flags if flag.startswith('-l')]
            return include_dirs, library_dirs, libraries
    except FileNotFoundError:
        pass
    
    # Fallback to common paths
    common_includes = [
        "/usr/local/include/opencv4",
        "/opt/homebrew/include/opencv4", 
        "/usr/include/opencv4"
    ]
    
    common_libs = [
        "/usr/local/lib",
        "/opt/homebrew/lib",
        "/usr/lib"
    ]
    
    for inc_path in common_includes:
        if Path(inc_path).exists():
            return [inc_path], common_libs, ["opencv_core", "opencv_imgproc", "opencv_imgcodecs"]
    
    return [], [], []

def create_extension():
    """Create the pybind11 extension."""
    # Check dependencies
    if not check_opencv():
        sys.exit(1)
    
    onnx_include, onnx_lib = find_onnxruntime()
    if not onnx_include or not onnx_lib:
        sys.exit(1)
    
    # Get OpenCV configuration
    opencv_includes, opencv_lib_dirs, opencv_libs = get_opencv_flags()
    
    # Define extension
    ext_modules = [
        Pybind11Extension(
            "cpp_geoguessr",
            [
                str(project_dir / "src" / "predictor.cpp"),
                str(project_dir / "src" / "bindings.cpp"),
            ],
            include_dirs=[
                str(project_dir / "src"),
                onnx_include,
            ] + opencv_includes,
            library_dirs=[
                onnx_lib,
            ] + opencv_lib_dirs,
            libraries=[
                "onnxruntime",
            ] + opencv_libs,
            cxx_std=17,
            define_macros=[("VERSION_INFO", f'"{__version__}"')],
        ),
    ]
    
    return ext_modules

class CustomBuildExt(build_ext):
    """Custom build extension to handle platform-specific settings."""
    
    def build_extensions(self):
        # Platform-specific compiler flags
        if sys.platform == "darwin":  # macOS
            for ext in self.extensions:
                ext.extra_compile_args.append("-std=c++17")
                ext.extra_compile_args.append("-O3")
                ext.extra_link_args.append("-Wl,-rpath,@loader_path")
        elif sys.platform.startswith("linux"):  # Linux
            for ext in self.extensions:
                ext.extra_compile_args.append("-std=c++17")
                ext.extra_compile_args.append("-O3")
                ext.extra_compile_args.append("-march=native")
                ext.extra_link_args.append("-Wl,-rpath,$ORIGIN")
        
        super().build_extensions()

if __name__ == "__main__":
    print("🚀 Building High-Performance C++ GeoGuessr Inference Engine")
    print("=" * 60)
    
    ext_modules = create_extension()
    
    setup(
        name="cpp_geoguessr",
        version=__version__,
        author="GeoGuessr AI Team",
        description="High-performance C++ inference engine for GeoGuessr predictions",
        long_description="""
        This module provides a high-performance C++ implementation for GeoGuessr country
        prediction using ONNX models. It replaces the Python-based inference pipeline
        with optimized C++ code using OpenCV for image processing and ONNX Runtime for
        model inference.
        """,
        ext_modules=ext_modules,
        cmdclass={"build_ext": CustomBuildExt},
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=[
            "opencv-python>=4.5.0",
        ],
    )