"""
High-Performance Flask App with C++ Inference Engine Integration

This enhanced version of the Flask app integrates the C++ inference engine
for maximum performance while maintaining compatibility with the existing
Python-based CLIP inference as a fallback.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import time
from pathlib import Path

# Try to import the C++ inference engine
CPP_AVAILABLE = False
try:
    # Add the cpp_inference directory to the Python path
    cpp_path = Path(__file__).parent / "cpp_inference"
    if cpp_path.exists():
        sys.path.insert(0, str(cpp_path))
    
    import cpp_geoguessr
    CPP_AVAILABLE = True
    print("🚀 High-performance C++ inference engine loaded successfully!")
    print(f"   Module version: {cpp_geoguessr.get_version()}")
    print(f"   Available countries: {len(cpp_geoguessr.Predictor('/tmp/dummy.onnx' if os.path.exists('/tmp/dummy.onnx') else 'dummy').get_countries()) if False else 'N/A (no model loaded)'}")
except ImportError as e:
    print("⚠️  C++ inference engine not available, falling back to Python implementation")
    print(f"   Error: {e}")
    print("   To enable C++ acceleration:")
    print("   1. cd backend/cpp_inference")
    print("   2. ./build.sh")

# Always import Python implementation as fallback
from ml_module import GeoGuessrPredictor

app = Flask(__name__)

# Allow CORS for specific origins
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Global predictor instance
predictor = None
cpp_predictor = None
inference_stats = {
    "total_predictions": 0,
    "cpp_predictions": 0,
    "python_predictions": 0,
    "average_cpp_time_ms": 0.0,
    "average_python_time_ms": 0.0,
}

def initialize_predictors():
    """Initialize both C++ and Python predictors."""
    global predictor, cpp_predictor
    
    if CPP_AVAILABLE:
        # Try to initialize C++ predictor
        try:
            # Look for ONNX model files
            model_paths = [
                "model/geoguessr_model.onnx",
                "../model/geoguessr_model.onnx", 
                "geoguessr_model.onnx",
                # Add more potential paths as needed
            ]
            
            onnx_model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    onnx_model_path = path
                    break
            
            if onnx_model_path:
                cpp_predictor = cpp_geoguessr.Predictor(onnx_model_path)
                print(f"✅ C++ predictor initialized with model: {onnx_model_path}")
            else:
                print("⚠️  No ONNX model found. C++ predictor not initialized.")
                print("   Expected locations:", model_paths)
                print("   You can still use the Python fallback.")
        except Exception as e:
            print(f"❌ Failed to initialize C++ predictor: {e}")
    
    # Always initialize Python predictor as fallback
    try:
        predictor = GeoGuessrPredictor()
        print("✅ Python predictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Python predictor: {e}")
        raise

# Initialize predictors
initialize_predictors()

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def convert_cpp_to_python_format(cpp_result):
    """Convert C++ prediction format to match Python format."""
    try:
        # C++ returns indices, we need to convert to country names
        if cpp_predictor and "predicted_regions" in cpp_result and "confidence_scores" in cpp_result:
            countries = cpp_predictor.get_countries()
            predicted_regions = []
            
            for idx in cpp_result["predicted_regions"]:
                if 0 <= int(idx) < len(countries):
                    predicted_regions.append(countries[int(idx)])
                else:
                    predicted_regions.append("Unknown")
            
            return {
                "predicted_regions": predicted_regions,
                "confidence_scores": cpp_result["confidence_scores"]
            }
    except Exception as e:
        print(f"Error converting C++ result: {e}")
    
    return cpp_result

def update_stats(prediction_time_ms, used_cpp=False):
    """Update performance statistics."""
    global inference_stats
    
    inference_stats["total_predictions"] += 1
    
    if used_cpp:
        inference_stats["cpp_predictions"] += 1
        # Running average
        n = inference_stats["cpp_predictions"]
        current_avg = inference_stats["average_cpp_time_ms"]
        inference_stats["average_cpp_time_ms"] = (current_avg * (n-1) + prediction_time_ms) / n
    else:
        inference_stats["python_predictions"] += 1
        # Running average  
        n = inference_stats["python_predictions"]
        current_avg = inference_stats["average_python_time_ms"]
        inference_stats["average_python_time_ms"] = (current_avg * (n-1) + prediction_time_ms) / n

@app.route('/predict', methods=['POST'])
def predict_country():
    """Enhanced prediction endpoint with C++ acceleration."""
    if 'image' not in request.files:
        print("No image file provided")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    try:
        start_time = time.time()
        used_cpp = False
        
        # Try C++ predictor first if available
        if CPP_AVAILABLE and cpp_predictor is not None and cpp_predictor.is_ready():
            try:
                print("🚀 Using high-performance C++ inference engine...")
                
                # Read image data directly from memory
                image_bytes = file.read()
                file.seek(0)  # Reset file pointer for fallback if needed
                
                # Run C++ prediction
                cpp_result = cpp_predictor.predict(image_bytes)
                
                # Convert format to match Python interface
                prediction = convert_cpp_to_python_format(cpp_result)
                used_cpp = True
                
                print(f"✅ C++ prediction completed")
                
            except Exception as e:
                print(f"⚠️  C++ prediction failed, falling back to Python: {e}")
                # Don't set cpp_predictor to None here - leave it for retry
        
        # Fallback to Python predictor
        if not used_cpp:
            if not predictor:
                return jsonify({'error': 'No predictor available'}), 500
            
            print("🐍 Using Python inference engine...")
            
            # Save the uploaded file for Python predictor
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            # Get prediction using Python
            prediction = predictor.predict(file_path)
            
            print(f"✅ Python prediction completed")
        
        # Calculate prediction time
        prediction_time_ms = (time.time() - start_time) * 1000
        update_stats(prediction_time_ms, used_cpp)
        
        print(f"⏱️  Prediction time: {prediction_time_ms:.1f}ms ({'C++' if used_cpp else 'Python'})")
        
        # Process results (same logic as original)
        if 'predicted_regions' in prediction and 'confidence_scores' in prediction:
            # Extract the top country and its confidence score
            top_country = prediction['predicted_regions'][0]
            top_percentage = prediction['confidence_scores'][0] * 100
            
            # Get the top 3 countries
            top_3_countries = prediction['predicted_regions'][:3]
            top_3_percentages = [score * 100 for score in prediction['confidence_scores'][:3]]

            # Enhanced response with performance info
            response_data = {
                'top_country': top_country,
                'top_percentage': top_percentage,
                'top_3_countries': top_3_countries,
                'top_3_percentages': top_3_percentages,
                'performance': {
                    'inference_engine': 'C++' if used_cpp else 'Python',
                    'prediction_time_ms': round(prediction_time_ms, 2),
                    'cpp_available': CPP_AVAILABLE and cpp_predictor is not None
                }
            }
            
            print("Response data:", {k: v for k, v in response_data.items() if k != 'performance'})
            return jsonify(response_data)
        else:
            print("Prediction did not contain expected keys.")
            return jsonify({'error': 'Prediction format is incorrect'}), 500

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get performance statistics."""
    return jsonify({
        'inference_stats': inference_stats,
        'system_info': {
            'cpp_available': CPP_AVAILABLE,
            'cpp_predictor_ready': cpp_predictor is not None and (cpp_predictor.is_ready() if cpp_predictor else False),
            'python_predictor_ready': predictor is not None,
        }
    })

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Benchmark both inference engines."""
    if not CPP_AVAILABLE or not cpp_predictor:
        return jsonify({'error': 'C++ predictor not available for benchmarking'}), 400
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    iterations = request.form.get('iterations', 10, type=int)
    
    try:
        image_bytes = file.read()
        
        # Benchmark C++ predictor
        cpp_benchmark = cpp_geoguessr.benchmark_prediction(cpp_predictor, image_bytes, iterations)
        
        # Benchmark Python predictor
        file.seek(0)
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        python_times = []
        for _ in range(iterations):
            start = time.time()
            predictor.predict(file_path)
            python_times.append((time.time() - start) * 1000)
        
        python_avg = sum(python_times) / len(python_times)
        
        return jsonify({
            'cpp_benchmark': dict(cpp_benchmark),
            'python_benchmark': {
                'average_time_ms': python_avg,
                'total_time_ms': sum(python_times),
                'iterations': iterations
            },
            'speedup': f"{python_avg / cpp_benchmark['average_time_ms']:.2f}x"
        })
        
    except Exception as e:
        return jsonify({'error': f'Benchmark failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌍 GeoGuessr AI - High-Performance Inference Server")
    print("="*60)
    print(f"🔧 C++ Acceleration: {'✅ Enabled' if CPP_AVAILABLE else '❌ Disabled'}")
    print(f"🐍 Python Fallback: {'✅ Ready' if predictor else '❌ Not Available'}")
    
    if CPP_AVAILABLE:
        print(f"⚡ C++ Predictor: {'✅ Ready' if cpp_predictor else '⚠️  No Model'}")
    
    print("\n📊 Available endpoints:")
    print("  POST /predict  - Main prediction endpoint") 
    print("  GET  /stats    - Performance statistics")
    print("  POST /benchmark - Benchmark both engines")
    print("="*60)
    
    app.run(port=8000, debug=True)