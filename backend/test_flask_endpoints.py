#!/usr/bin/env python3
"""
Test the Flask endpoints with actual HTTP requests using test images.
"""

import requests
import threading
import time
import sys
from pathlib import Path
import subprocess

def start_flask_server():
    """Start the Flask server in a subprocess."""
    try:
        # Start the Flask app
        process = subprocess.Popen(
            [sys.executable, 'app_cpp.py'],
            cwd='/Users/nikhiltiwari/Documents/CODE/hackathons/maddata/backend',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"Failed to start Flask server: {e}")
        return None

def wait_for_server(url="http://localhost:8000", timeout=30):
    """Wait for the Flask server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/stats", timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def test_endpoints():
    """Test the Flask endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Flask endpoints...")
    
    # Test 1: Stats endpoint
    try:
        print("\n📊 Testing /stats endpoint...")
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Stats endpoint working")
            print(f"   C++ Available: {stats.get('system_info', {}).get('cpp_available', False)}")
            print(f"   Python Ready: {stats.get('system_info', {}).get('python_predictor_ready', False)}")
            print(f"   Total Predictions: {stats.get('inference_stats', {}).get('total_predictions', 0)}")
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats endpoint error: {e}")
    
    # Test 2: Prediction endpoint with test images
    test_images_dir = Path("/Users/nikhiltiwari/Documents/CODE/hackathons/maddata/backend/testing_images")
    test_images = list(test_images_dir.glob("*.png"))[:2]  # Test 2 images
    
    print(f"\\n🖼️  Testing /predict endpoint with {len(test_images)} images...")
    
    results = []
    for i, image_path in enumerate(test_images):
        try:
            print(f"\\n   Testing {i+1}/{len(test_images)}: {image_path.name}")
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{base_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Prediction: {data.get('top_country', 'Unknown')} ({data.get('top_percentage', 0):.1f}%)")
                
                if 'performance' in data:
                    perf = data['performance']
                    print(f"   ⚡ Engine: {perf.get('inference_engine', 'Unknown')}")
                    print(f"   ⏱️  Time: {perf.get('prediction_time_ms', 0):.1f}ms")
                
                results.append({
                    'image': image_path.name,
                    'prediction': data.get('top_country', 'Unknown'),
                    'confidence': data.get('top_percentage', 0),
                    'engine': data.get('performance', {}).get('inference_engine', 'Unknown'),
                    'time_ms': data.get('performance', {}).get('prediction_time_ms', 0)
                })
            else:
                print(f"   ❌ Prediction failed: {response.status_code}")
                print(f"      Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error testing {image_path.name}: {e}")
    
    # Test 3: Stats after predictions
    try:
        print(f"\\n📊 Testing /stats after predictions...")
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            inference_stats = stats.get('inference_stats', {})
            print("✅ Updated stats retrieved")
            print(f"   Total Predictions: {inference_stats.get('total_predictions', 0)}")
            print(f"   Python Predictions: {inference_stats.get('python_predictions', 0)}")
            print(f"   C++ Predictions: {inference_stats.get('cpp_predictions', 0)}")
            print(f"   Avg Python Time: {inference_stats.get('average_python_time_ms', 0):.1f}ms")
    except Exception as e:
        print(f"❌ Updated stats error: {e}")
    
    # Summary
    if results:
        print(f"\\n🎯 Prediction Results Summary:")
        for result in results:
            print(f"   {result['image']}: {result['prediction']} ({result['confidence']:.1f}%) - {result['engine']} ({result['time_ms']:.1f}ms)")
    
    return results

def run_flask_test():
    """Run the complete Flask test."""
    print("🌐 GeoGuessr Flask App - Live Testing")
    print("=" * 50)
    
    # Start Flask server
    print("🚀 Starting Flask server...")
    flask_process = start_flask_server()
    
    if not flask_process:
        print("❌ Failed to start Flask server")
        return
    
    try:
        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        if wait_for_server():
            print("✅ Flask server is ready!")
            
            # Run tests
            results = test_endpoints()
            
            print(f"\\n🎉 Flask testing completed!")
            if results:
                avg_time = sum(r['time_ms'] for r in results) / len(results)
                print(f"   Average response time: {avg_time:.1f}ms")
                print(f"   Images tested: {len(results)}")
        else:
            print("❌ Server failed to start within timeout")
            
    except KeyboardInterrupt:
        print("\\n⚠️  Test interrupted by user")
    finally:
        # Clean up
        print("🧹 Shutting down Flask server...")
        flask_process.terminate()
        flask_process.wait(timeout=5)
        print("✅ Flask server shut down")

if __name__ == "__main__":
    run_flask_test()