#!/bin/bash

# High-Performance C++ GeoGuessr Inference Engine Build Script
# This script automates the building process for the C++ inference module

set -e  # Exit on any error

echo "🚀 Building High-Performance C++ GeoGuessr Inference Engine"
echo "==========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="$SCRIPT_DIR/build"

echo -e "${BLUE}📁 Project directory: $SCRIPT_DIR${NC}"

# Check for required tools
echo -e "\n${BLUE}🔍 Checking prerequisites...${NC}"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_dependency() {
    if command_exists "$1"; then
        echo -e "${GREEN}✅ $1 found${NC}"
        return 0
    else
        echo -e "${RED}❌ $1 not found${NC}"
        return 1
    fi
}

# Check essential tools
MISSING_DEPS=0

if ! check_dependency "cmake"; then
    echo -e "${YELLOW}Install CMake: https://cmake.org/download/${NC}"
    MISSING_DEPS=1
fi

if ! check_dependency "python3"; then
    echo -e "${YELLOW}Install Python 3: https://python.org${NC}"
    MISSING_DEPS=1
fi

if ! check_dependency "g++" && ! command_exists "clang++"; then
    echo -e "${RED}❌ No C++ compiler found (g++ or clang++)${NC}"
    echo -e "${YELLOW}Install build tools for your system${NC}"
    MISSING_DEPS=1
else
    echo -e "${GREEN}✅ C++ compiler found${NC}"
fi

# Check for OpenCV
echo -e "\n${BLUE}🔍 Checking for OpenCV...${NC}"
if pkg-config --exists opencv4 2>/dev/null; then
    echo -e "${GREEN}✅ OpenCV 4 found: $(pkg-config --modversion opencv4)${NC}"
elif pkg-config --exists opencv 2>/dev/null; then
    echo -e "${GREEN}✅ OpenCV found: $(pkg-config --modversion opencv)${NC}"
else
    echo -e "${RED}❌ OpenCV not found${NC}"
    echo -e "${YELLOW}Install OpenCV:${NC}"
    echo -e "  ${YELLOW}macOS: brew install opencv${NC}"
    echo -e "  ${YELLOW}Ubuntu: sudo apt-get install libopencv-dev${NC}"
    MISSING_DEPS=1
fi

# Check for ONNX Runtime
echo -e "\n${BLUE}🔍 Checking for ONNX Runtime...${NC}"
ONNX_FOUND=0

# Check common locations
if [[ -f "/usr/local/include/onnxruntime/onnxruntime_cxx_api.h" ]] && [[ -f "/usr/local/lib/libonnxruntime.so" || -f "/usr/local/lib/libonnxruntime.dylib" ]]; then
    echo -e "${GREEN}✅ ONNX Runtime found in /usr/local/${NC}"
    ONNX_FOUND=1
elif [[ -f "/opt/homebrew/include/onnxruntime/onnxruntime_cxx_api.h" ]] && [[ -f "/opt/homebrew/lib/libonnxruntime.dylib" ]]; then
    echo -e "${GREEN}✅ ONNX Runtime found in /opt/homebrew/${NC}"
    ONNX_FOUND=1
fi

if [[ $ONNX_FOUND -eq 0 ]]; then
    echo -e "${RED}❌ ONNX Runtime not found${NC}"
    echo -e "${YELLOW}Install ONNX Runtime:${NC}"
    echo -e "  ${YELLOW}macOS: brew install onnxruntime${NC}"
    echo -e "  ${YELLOW}Linux: Download from https://github.com/microsoft/onnxruntime/releases${NC}"
    echo -e "  ${YELLOW}       Extract to /usr/local/ (requires sudo)${NC}"
    MISSING_DEPS=1
fi

if [[ $MISSING_DEPS -eq 1 ]]; then
    echo -e "\n${RED}❌ Missing dependencies. Please install them and try again.${NC}"
    exit 1
fi

echo -e "\n${GREEN}✅ All prerequisites satisfied!${NC}"

# Clean previous build
echo -e "\n${BLUE}🧹 Cleaning previous build...${NC}"
if [[ -d "$BUILD_DIR" ]]; then
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}✅ Cleaned build directory${NC}"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "\n${BLUE}⚙️  Configuring build with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo -e "\n${BLUE}🔨 Building C++ inference engine...${NC}"
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if ls cpp_geoguessr*.so 1> /dev/null 2>&1; then
    echo -e "\n${GREEN}🎉 Build successful!${NC}"
    
    # Copy the built module to the parent directory for easy access
    cp cpp_geoguessr* ../
    echo -e "${GREEN}✅ Module copied to cpp_inference directory${NC}"
    
    echo -e "\n${GREEN}📦 Built files:${NC}"
    ls -la cpp_geoguessr*
    
    # Test import (optional)
    echo -e "\n${BLUE}🧪 Testing Python import...${NC}"
    cd "$SCRIPT_DIR"
    if python3 -c "import cpp_geoguessr; print('✅ Module imports successfully')" 2>/dev/null; then
        echo -e "${GREEN}✅ Python module import test passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Module built but import test failed. Check dependencies.${NC}"
    fi
    
else
    echo -e "\n${RED}❌ Build failed!${NC}"
    echo -e "${YELLOW}Check the error messages above for details.${NC}"
    exit 1
fi

echo -e "\n${GREEN}🚀 C++ inference engine is ready to use!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Update your Flask app to import: ${YELLOW}from cpp_geoguessr import Predictor${NC}"
echo -e "2. Replace the Python inference with: ${YELLOW}predictor = Predictor('path/to/model.onnx')${NC}"
echo -e "3. Call predictions with: ${YELLOW}predictor.predict(image_bytes)${NC}"