from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_module import GeoGuessrPredictor
import os

app = Flask(__name__)

# Allow CORS for specific origins
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Initialize the predictor
predictor = GeoGuessrPredictor()

UPLOAD_FOLDER = 'uploads'  # Create an uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
# @requires_auth  # Example of a decorator that might restrict access
def predict_country():
    if 'image' not in request.files:
        print("No image file provided")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Get prediction
        prediction = predictor.predict(file_path)
        print("Prediction successful:", prediction)  # Log the prediction
        
        # Check if the prediction contains the expected keys
        if 'predicted_regions' in prediction and 'confidence_scores' in prediction:
            # Extract the top country and its confidence score
            top_country = prediction['predicted_regions'][0]
            top_percentage = prediction['confidence_scores'][0] * 100  # Convert to percentage
            
            # Get the top 3 countries
            top_3_countries = prediction['predicted_regions'][:3]
            top_3_percentages = [score * 100 for score in prediction['confidence_scores'][:3]]  # Convert to percentage

            # Log the data being returned
            response_data = {
                'top_country': top_country,
                'top_percentage': top_percentage,
                'top_3_countries': top_3_countries,
                'top_3_percentages': top_3_percentages
            }
            print("Response data:", response_data)  # Log the response data

            return jsonify(response_data)
        else:
            print("Prediction did not contain expected keys.")
            return jsonify({'error': 'Prediction format is incorrect'}), 500

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True)