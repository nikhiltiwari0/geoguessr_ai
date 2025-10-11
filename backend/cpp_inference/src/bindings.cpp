#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "predictor.h"
#include <fstream>
#include <chrono>

namespace py = pybind11;

/**
 * Python bindings for the GeoGuessrPredictor C++ class.
 * This module exposes the high-performance C++ inference engine to Python,
 * allowing seamless integration with the Flask application.
 */

PYBIND11_MODULE(cpp_geoguessr, m) {
    m.doc() = "High-performance C++ inference module for GeoGuessr AI prediction";
    
    // Bind the main predictor class
    py::class_<GeoGuessrPredictor>(m, "Predictor")
        .def(py::init<const std::string&>(), 
             py::arg("model_path"),
             "Initialize the GeoGuessr predictor with an ONNX model")
        
        .def(py::init<const std::string&, int>(), 
             py::arg("model_path"), py::arg("input_size"),
             "Initialize the GeoGuessr predictor with an ONNX model and custom input size")
        
        .def("predict", [](GeoGuessrPredictor &self, const py::bytes &image_bytes) {
            // Convert Python bytes to std::vector<unsigned char>
            std::string bytes_str = image_bytes;
            std::vector<unsigned char> vec(bytes_str.begin(), bytes_str.end());
            return self.predict(vec);
        }, py::arg("image_bytes"),
           "Run inference on raw image bytes and return country predictions")
        
        .def("predict_from_file", &GeoGuessrPredictor::predict_from_file,
             py::arg("image_path"),
             "Run inference on an image file and return country predictions")
        
        .def("get_countries", &GeoGuessrPredictor::get_countries,
             "Get the list of supported countries")
        
        .def("is_ready", &GeoGuessrPredictor::is_ready,
             "Check if the model is loaded and ready for inference")
        
        .def("__repr__", [](const GeoGuessrPredictor &predictor) {
            return "<GeoGuessrPredictor: " + std::to_string(predictor.get_countries().size()) + " countries>";
        });
    
    // Helper function to convert image file to bytes (useful for testing)
    m.def("load_image_bytes", [](const std::string& image_path) {
        std::ifstream file(image_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open image file: " + image_path);
        }
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            throw std::runtime_error("Error reading image file: " + image_path);
        }
        
        return py::bytes(buffer.data(), size);
    }, py::arg("image_path"), "Load an image file as bytes for testing purposes");
    
    // Utility function to get version info
    m.def("get_version", []() {
        return "1.0.0";
    }, "Get the version of the C++ inference module");
    
    // Performance benchmarking helper
    m.def("benchmark_prediction", [](GeoGuessrPredictor &predictor, const py::bytes &image_bytes, int num_iterations) {
        std::string bytes_str = image_bytes;
        std::vector<unsigned char> vec(bytes_str.begin(), bytes_str.end());
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::unordered_map<std::string, std::vector<float>> result;
        for (int i = 0; i < num_iterations; ++i) {
            result = predictor.predict(vec);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double avg_time_ms = static_cast<double>(duration.count()) / num_iterations;
        
        py::dict benchmark_result;
        benchmark_result["average_time_ms"] = avg_time_ms;
        benchmark_result["total_time_ms"] = duration.count();
        benchmark_result["iterations"] = num_iterations;
        benchmark_result["last_prediction"] = result;
        
        return benchmark_result;
    }, py::arg("predictor"), py::arg("image_bytes"), py::arg("num_iterations") = 10,
       "Benchmark the prediction performance");
}
