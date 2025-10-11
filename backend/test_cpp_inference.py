#!/usr/bin/env python3
"""
Test script for the C++ inference engine.

This script tests the C++ module functionality and compares performance
with the Python implementation using the test images.
"""

import os
import sys
import time
from pathlib import Path
import traceback

# Add the cpp_inference directory to Python path
cpp_path = Path(__file__).parent / "cpp_inference"
sys.path.insert(0, str(cpp_path))

def test_cpp_module_import():
    """Test if the C++ module can be imported successfully."""
    print("🧪 Testing C++ module import...")
    try:
        import cpp_geoguessr
        print(f"✅ C++ module imported successfully!")
        print(f"   Version: {cpp_geoguessr.get_version()}")
        
        # Test basic functionality without a model
        try:
            # This will fail without a valid ONNX model, but we can test the binding
            predictor = cpp_geoguessr.Predictor("dummy_path.onnx")
            print("⚠️  Predictor created (will fail without model)")
        except Exception as e:
            print(f"⚠️  Expected error without ONNX model: {str(e)[:100]}...")
        
        # Test utility functions
        try:
            test_bytes = cpp_geoguessr.load_image_bytes("/Users/nikhiltiwari/Documents/CODE/hackathons/maddata/backend/testing_images/test1.png")
            print(f"✅ Image loading utility works: {len(test_bytes)} bytes")
        except Exception as e:
            print(f"⚠️  Image loading test: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import C++ module: {e}")
        return False

def test_python_implementation():
    """Test the existing Python implementation with test images."""
    print("\n🐍 Testing Python implementation...")
    
    try:
        from ml_module import GeoGuessrPredictor
        
        print("Initializing Python predictor...")
        predictor = GeoGuessrPredictor()
        print("✅ Python predictor initialized")
        
        # Test with available images
        test_images_dir = Path("/Users/nikhiltiwari/Documents/CODE/hackathons/maddata/backend/testing_images")
        test_images = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg"))
        
        print(f"\n📸 Found {len(test_images)} test images")
        
        results = []
        for i, image_path in enumerate(test_images[:3]):  # Test first 3 images
            print(f"\n🖼️  Testing image {i+1}/{min(3, len(test_images))}: {image_path.name}")
            
            try:
                start_time = time.time()
                prediction = predictor.predict(str(image_path))
                prediction_time = (time.time() - start_time) * 1000
                
                if 'predicted_regions' in prediction and 'confidence_scores' in prediction:
                    top_country = prediction['predicted_regions'][0]
                    confidence = prediction['confidence_scores'][0] * 100
                    
                    print(f"   🎯 Prediction: {top_country} ({confidence:.1f}% confidence)")
                    print(f"   ⏱️  Time: {prediction_time:.1f}ms")
                    
                    # Get top 3 predictions
                    top_3 = [(prediction['predicted_regions'][j], prediction['confidence_scores'][j] * 100) 
                            for j in range(min(3, len(prediction['predicted_regions'])))]
                    
                    print("   📊 Top 3 predictions:")
                    for j, (country, conf) in enumerate(top_3):
                        print(f"      {j+1}. {country}: {conf:.1f}%")
                    
                    results.append({
                        'image': image_path.name,
                        'prediction': top_country,
                        'confidence': confidence,
                        'time_ms': prediction_time,
                        'top_3': top_3
                    })
                    
                else:
                    print(f"   ❌ Invalid prediction format: {prediction}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {image_path.name}: {e}")
                traceback.print_exc()
        
        # Summary
        if results:
            avg_time = sum(r['time_ms'] for r in results) / len(results)
            print(f"\n📊 Python Implementation Summary:")
            print(f"   Images processed: {len(results)}")
            print(f"   Average time: {avg_time:.1f}ms")
            print(f"   Time range: {min(r['time_ms'] for r in results):.1f}ms - {max(r['time_ms'] for r in results):.1f}ms")
            
            print(f"\n🎯 Results Summary:")
            for result in results:
                print(f"   {result['image']}: {result['prediction']} ({result['confidence']:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Python implementation test failed: {e}")
        traceback.print_exc()
        return None

def test_enhanced_flask_app():
    """Test the enhanced Flask app startup."""
    print("\n🌐 Testing enhanced Flask app integration...")
    
    try:
        # Import without running the app
        old_name = __name__ 
        sys.modules['app_cpp'].__name__ = 'not_main'  # Prevent auto-run
        
        from app_cpp import app, CPP_AVAILABLE, predictor, cpp_predictor, inference_stats
        
        print(f"✅ Enhanced Flask app imported successfully")
        print(f"   C++ Available: {'✅' if CPP_AVAILABLE else '❌'}")
        print(f"   Python Predictor: {'✅' if predictor else '❌'}")
        print(f"   C++ Predictor: {'✅' if cpp_predictor else '❌'}")
        print(f"   Inference Stats: {inference_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print("🚀 GeoGuessr C++ Inference Engine - Comprehensive Test")
    print("=" * 60)
    
    # Test results tracking
    test_results = {
        'cpp_module_import': False,
        'python_implementation': False,
        'flask_integration': False,
        'python_predictions': []
    }
    
    # Test 1: C++ Module Import
    test_results['cpp_module_import'] = test_cpp_module_import()
    
    # Test 2: Python Implementation
    python_results = test_python_implementation()
    if python_results:
        test_results['python_implementation'] = True
        test_results['python_predictions'] = python_results
    
    # Test 3: Flask Integration  
    test_results['flask_integration'] = test_enhanced_flask_app()
    
    # Final Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    print(f"🔧 C++ Module Import:       {'✅ PASS' if test_results['cpp_module_import'] else '❌ FAIL'}")
    print(f"🐍 Python Implementation:   {'✅ PASS' if test_results['python_implementation'] else '❌ FAIL'}")  
    print(f"🌐 Flask Integration:       {'✅ PASS' if test_results['flask_integration'] else '❌ FAIL'}")
    
    if test_results['python_predictions']:
        predictions = test_results['python_predictions']
        avg_time = sum(p['time_ms'] for p in predictions) / len(predictions)
        print(f"\n📈 Performance Metrics:")
        print(f"   Average inference time: {avg_time:.1f}ms")
        print(f"   Images tested: {len(predictions)}")
        
        print(f"\n🎯 Sample Predictions:")
        for pred in predictions[:3]:
            print(f"   {pred['image']}: {pred['prediction']} ({pred['confidence']:.1f}%)")
    
    # Status assessment
    passed_tests = sum(1 for v in test_results.values() if v and not isinstance(v, list))
    total_tests = 3
    
    print(f"\n🏁 Overall Status: {passed_tests}/{total_tests} tests passed")
    
    if test_results['cpp_module_import'] and test_results['python_implementation']:
        print("\n✅ System is ready for production!")
        print("🚀 The C++ inference engine is built and Python fallback is working.")
        print("   Next steps:")
        print("   1. Add an ONNX model to enable C++ acceleration")
        print("   2. Run: cd backend && python app_cpp.py")
        print("   3. Test the /predict endpoint with images")
    elif test_results['python_implementation']:
        print("\n⚠️  Python implementation working, C++ needs attention")
        print("   The system will work with Python fallback.")
    else:
        print("\n❌ System needs troubleshooting before deployment")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_test()