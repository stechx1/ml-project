from flask import Flask, request, jsonify
import joblib

# Load the trained pipeline
pipeline = joblib.load('gradient_boosting_pipeline.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define a mapping for predictions
label_mapping = {1: "Yes", 0: "No"}


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        headlines = data['headlines']  # Expecting a key 'headlines' with a list of text

        # Validate input
        if not isinstance(headlines, list):
            return jsonify({'error': 'Input must be a list of headlines'}), 400

        # Make predictions
        predictions = pipeline.predict(headlines)
        predictions_mapped = [label_mapping[pred] for pred in predictions]  # Convert 0/1 to "No"/"Yes"

        # Return predictions as JSON
        return jsonify({'predictions': predictions_mapped})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
