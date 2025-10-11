#include "predictor.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <regex>

GeoGuessrPredictor::GeoGuessrPredictor(const std::string& model_path, int input_size)
    : input_size_(input_size), model_loaded_(false) {
    
    try {
        // Initialize ONNX Runtime
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GeoGuessrInference");
        allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        
        // Initialize country data
        initialize_countries();
        initialize_domain_indicators();
        
        // Load the ONNX model
        load_model(model_path);
        
        std::cout << "GeoGuessrPredictor initialized successfully with " 
                  << countries_.size() << " countries" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing GeoGuessrPredictor: " << e.what() << std::endl;
        throw;
    }
}

GeoGuessrPredictor::~GeoGuessrPredictor() = default;

void GeoGuessrPredictor::load_model(const std::string& model_path) {
    try {
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load the model
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        // Get input/output information
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();
        
        if (num_input_nodes == 0 || num_output_nodes == 0) {
            throw std::runtime_error("Model has no input or output nodes");
        }
        
        // Get input node information
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        
        input_node_dims_ = input_tensor_info.GetShape();
        
        // Handle dynamic batch size
        if (input_node_dims_[0] == -1) {
            input_node_dims_[0] = 1; // Set batch size to 1
        }
        
        // Verify input dimensions match expected format [batch, channels, height, width]
        if (input_node_dims_.size() != 4) {
            throw std::runtime_error("Expected 4D input tensor [batch, channels, height, width]");
        }
        
        // Update input size based on model requirements
        if (input_node_dims_[2] != input_size_ || input_node_dims_[3] != input_size_) {
            input_size_ = static_cast<int>(input_node_dims_[2]);
            std::cout << "Updated input size to match model: " << input_size_ << "x" << input_size_ << std::endl;
        }
        
        // Store node names (we'll allocate these properly)
        input_node_names_.clear();
        output_node_names_.clear();
        
        // Get input node name
        auto input_name = session_->GetInputNameAllocated(0, *allocator_);
        input_node_names_.push_back(input_name.get());
        
        // Get output node name
        auto output_name = session_->GetOutputNameAllocated(0, *allocator_);
        output_node_names_.push_back(output_name.get());
        
        model_loaded_ = true;
        std::cout << "Model loaded successfully. Input shape: [";
        for (size_t i = 0; i < input_node_dims_.size(); ++i) {
            std::cout << input_node_dims_[i];
            if (i < input_node_dims_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> GeoGuessrPredictor::predict(const std::vector<unsigned char>& image_bytes) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    try {
        // Decode image from bytes
        cv::Mat image = cv::imdecode(image_bytes, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to decode image from bytes");
        }
        
        return predict_from_mat(image);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in predict: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> GeoGuessrPredictor::predict_from_file(const std::string& image_path) {
    if (!model_loaded_) {
        throw std::runtime_error("Model not loaded");
    }
    
    try {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image from: " + image_path);
        }
        
        return predict_from_mat(image);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in predict_from_file: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> GeoGuessrPredictor::predict_from_mat(const cv::Mat& image) {
    // Preprocess the image
    cv::Mat processed_image = preprocess_image(image);
    
    // Extract domain hints from OCR (optional enhancement)
    auto domain_hints = extract_domain_hints(image);
    
    // Run inference
    std::vector<float> raw_predictions = run_inference(processed_image);
    
    // Process predictions and apply domain hints
    auto result = process_predictions(raw_predictions);
    
    // Apply domain hints to boost certain countries
    if (!domain_hints.empty()) {
        auto& confidence_scores = result["confidence_scores"];
        auto& predicted_regions = result["predicted_regions"];
        
        for (const auto& domain_hint : domain_hints) {
            const std::string& country = domain_hint.first;
            float boost = domain_hint.second;
            
            // Find the country in our predictions
            auto it = std::find_if(predicted_regions.begin(), predicted_regions.end(),
                [&country](float val) {
                    // This is a simplified lookup - in practice you'd maintain country-index mapping
                    return false; // Placeholder - would need proper country lookup
                });
        }
    }
    
    return result;
}

cv::Mat GeoGuessrPredictor::preprocess_image(const cv::Mat& image) {
    cv::Mat processed;
    
    // 1. Resize to model input size
    cv::resize(image, processed, cv::Size(input_size_, input_size_), 0, 0, cv::INTER_LINEAR);
    
    // 2. Convert BGR to RGB (OpenCV uses BGR by default)
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    
    // 3. Convert to float32 and normalize to [0, 1]
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    
    // 4. Apply ImageNet normalization (subtract mean, divide by std)
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }
    
    cv::merge(channels, processed);
    
    return processed;
}

std::vector<float> GeoGuessrPredictor::run_inference(const cv::Mat& processed_image) {
    try {
        // Create input tensor
        size_t input_tensor_size = processed_image.total() * processed_image.channels();
        
        // Convert CHW format (Channels, Height, Width) as expected by most models
        std::vector<float> input_data(input_tensor_size);
        
        // Convert HWC to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(processed_image, channels);
        
        size_t channel_size = input_size_ * input_size_;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_data.data() + c * channel_size, 
                       channels[c].ptr<float>(), 
                       channel_size * sizeof(float));
        }
        
        // Create ONNX tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault), 
            input_data.data(), 
            input_tensor_size,
            input_node_dims_.data(), 
            input_node_dims_.size()
        );
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_node_names_.data(), 
            &input_tensor, 
            1,
            output_node_names_.data(), 
            1
        );
        
        // Extract results
        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        
        size_t output_size = 1;
        for (auto dim : output_shape) {
            output_size *= dim;
        }
        
        return std::vector<float>(output_data, output_data + output_size);
    }
    catch (const std::exception& e) {
        std::cerr << "Error in run_inference: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> GeoGuessrPredictor::process_predictions(const std::vector<float>& raw_predictions) {
    if (raw_predictions.size() != countries_.size()) {
        std::cerr << "Warning: prediction size (" << raw_predictions.size() 
                  << ") doesn't match number of countries (" << countries_.size() << ")" << std::endl;
    }
    
    // Create country-score pairs
    std::vector<std::pair<std::string, float>> country_scores;
    size_t max_countries = std::min(raw_predictions.size(), countries_.size());
    
    for (size_t i = 0; i < max_countries; ++i) {
        country_scores.emplace_back(countries_[i], raw_predictions[i]);
    }
    
    // Sort by confidence score (descending)
    std::sort(country_scores.begin(), country_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Extract top 3 predictions
    std::vector<float> predicted_regions_indices;
    std::vector<float> confidence_scores;
    
    size_t top_k = std::min(size_t(3), country_scores.size());
    for (size_t i = 0; i < top_k; ++i) {
        predicted_regions_indices.push_back(static_cast<float>(i)); // Index in sorted order
        confidence_scores.push_back(country_scores[i].second);
    }
    
    // Convert country names to indices for compatibility with Python interface
    std::vector<float> country_indices;
    for (size_t i = 0; i < top_k; ++i) {
        // Find original index of country
        auto it = std::find(countries_.begin(), countries_.end(), country_scores[i].first);
        if (it != countries_.end()) {
            country_indices.push_back(static_cast<float>(std::distance(countries_.begin(), it)));
        } else {
            country_indices.push_back(-1.0f); // Error case
        }
    }
    
    // Return results in format compatible with Python code
    std::unordered_map<std::string, std::vector<float>> result;
    result["predicted_regions"] = country_indices;
    result["confidence_scores"] = confidence_scores;
    
    return result;
}

std::unordered_map<std::string, float> GeoGuessrPredictor::extract_domain_hints(const cv::Mat& image) {
    std::unordered_map<std::string, float> hints;
    
    // This is a placeholder for OCR-based domain extraction
    // In a full implementation, you would:
    // 1. Use Tesseract OCR to extract text from the image
    // 2. Search for domain patterns (e.g., .fr, .de, .uk, etc.)
    // 3. Map domains to countries using domain_indicators_
    // 4. Return confidence boosts for detected countries
    
    // For now, return empty hints
    return hints;
}

void GeoGuessrPredictor::initialize_countries() {
    countries_ = {
        "Albania", "Andorra", "Argentina", "Australia", "Austria", "Bangladesh",
        "Belgium", "Bermuda", "Bhutan", "Bolivia", "Botswana", "Brazil", "Bulgaria",
        "Cambodia", "Canada", "Chile", "China", "Colombia", "Croatia", "Czechia",
        "Denmark", "Dominican Republic", "Ecuador", "Egypt", "Estonia", "Eswatini",
        "Faroe Islands", "Finland", "France", "Germany", "Ghana", "Greece", "Greenland",
        "Guatemala", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Ireland",
        "Israel", "Italy", "Japan", "Jordan", "Kenya", "Kyrgyzstan", "Laos", "Latvia",
        "Lesotho", "Lithuania", "Luxembourg", "Malaysia", "Mexico", "Mongolia",
        "Montenegro", "Nepal", "Netherlands", "New Zealand", "Nigeria", "North Macedonia",
        "North Mariana Islands", "Norway", "Pakistan", "Palestine", "Panama", "Peru",
        "Philippines", "Poland", "Portugal", "Puerto Rico", "Réunion", "Romania",
        "Russia", "Rwanda", "Senegal", "Serbia", "Singapore", "Slovakia", "Slovenia",
        "South Africa", "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland",
        "Taiwan", "Thailand", "Turkey", "Uganda", "Ukraine", "United Arab Emirates",
        "United Kingdom", "United States", "Uruguay", "Vietnam"
    };
}

void GeoGuessrPredictor::initialize_domain_indicators() {
    domain_indicators_ = {
        {".fr", "France"}, {".de", "Germany"}, {".uk", "United Kingdom"}, {".es", "Spain"},
        {".it", "Italy"}, {".pl", "Poland"}, {".nl", "Netherlands"}, {".be", "Belgium"},
        {".at", "Austria"}, {".ch", "Switzerland"}, {".us", "United States"}, {".ca", "Canada"},
        {".au", "Australia"}, {".nz", "New Zealand"}, {".sg", "Singapore"}, {".hk", "Hong Kong"},
        {".jp", "Japan"}, {".kr", "South Korea"}, {".tw", "Taiwan"}, {".th", "Thailand"},
        {".tr", "Turkey"}, {".ua", "Ukraine"}, {".ru", "Russia"}, {".br", "Brazil"},
        {".mx", "Mexico"}, {".ar", "Argentina"}, {".cl", "Chile"}, {".co", "Colombia"},
        {".pe", "Peru"}, {".za", "South Africa"}, {".eg", "Egypt"}, {".ng", "Nigeria"},
        {".ke", "Kenya"}, {".ug", "Uganda"}, {".in", "India"}, {".cn", "China"},
        {".id", "Indonesia"}, {".my", "Malaysia"}, {".ph", "Philippines"}, {".vn", "Vietnam"},
        {".lk", "Sri Lanka"}, {".ae", "United Arab Emirates"}, {".il", "Israel"},
        {".pk", "Pakistan"}, {".dk", "Denmark"}, {".fi", "Finland"}, {".no", "Norway"},
        {".se", "Sweden"}, {".pt", "Portugal"}, {".gr", "Greece"}, {".ie", "Ireland"},
        {".cz", "Czech Republic"}, {".hu", "Hungary"}, {".ro", "Romania"}, {".bg", "Bulgaria"},
        {".hr", "Croatia"}, {".rs", "Serbia"}, {".sk", "Slovakia"}, {".si", "Slovenia"},
        {".ee", "Estonia"}, {".lv", "Latvia"}, {".lt", "Lithuania"}, {".uy", "Uruguay"}
    };
}