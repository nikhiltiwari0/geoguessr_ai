#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declarations to avoid including heavy headers
namespace Ort {
    class Env;
    class Session;
    class AllocatorWithDefaultOptions;
    class Value;
}

/**
 * High-performance C++ GeoGuessr predictor that processes images and predicts countries.
 * This class replaces the Python-based CLIP inference with optimized C++ code using
 * OpenCV for image processing and ONNX Runtime for model inference.
 */
class GeoGuessrPredictor {
public:
    /**
     * Constructor - initializes the predictor with model path and configuration
     * @param model_path Path to the ONNX model file
     * @param input_size Expected input image size (default: 224x224)
     */
    explicit GeoGuessrPredictor(const std::string& model_path, int input_size = 224);
    
    /**
     * Destructor - cleans up ONNX resources
     */
    ~GeoGuessrPredictor();

    /**
     * Main prediction method - processes raw image bytes and returns country predictions
     * @param image_bytes Raw image data as bytes
     * @return Prediction results with countries and confidence scores
     */
    std::unordered_map<std::string, std::vector<float>> predict(const std::vector<unsigned char>& image_bytes);

    /**
     * Prediction method for file path input
     * @param image_path Path to image file
     * @return Prediction results with countries and confidence scores
     */
    std::unordered_map<std::string, std::vector<float>> predict_from_file(const std::string& image_path);

    /**
     * Get the list of supported countries
     * @return Vector of country names
     */
    const std::vector<std::string>& get_countries() const { return countries_; }

    /**
     * Check if the model is properly loaded
     * @return True if model is loaded and ready for inference
     */
    bool is_ready() const { return model_loaded_; }

private:
    // Core prediction pipeline methods
    std::unordered_map<std::string, std::vector<float>> predict_from_mat(const cv::Mat& image);
    cv::Mat preprocess_image(const cv::Mat& image);
    std::vector<float> run_inference(const cv::Mat& processed_image);
    std::unordered_map<std::string, std::vector<float>> process_predictions(const std::vector<float>& raw_predictions);
    
    // OCR-based domain detection for enhanced accuracy
    std::unordered_map<std::string, float> extract_domain_hints(const cv::Mat& image);
    
    // Utility methods
    void initialize_countries();
    void initialize_domain_indicators();
    void load_model(const std::string& model_path);
    std::unordered_map<std::string, std::vector<float>> create_mock_prediction(const cv::Mat& image);

    // Model and inference components (only initialized when model is loaded)
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;

    // Model metadata
    std::vector<int64_t> input_node_dims_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    
    // Configuration
    int input_size_;
    bool model_loaded_;
    
    // GeoGuessr data
    std::vector<std::string> countries_;
    std::unordered_map<std::string, std::string> domain_indicators_;
    
    // Image normalization parameters (ImageNet standard)
    const std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
};