from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    tv = float(request.form["tv"])
    radio = float(request.form["radio"])
    newspaper = float(request.form["newspaper"])

    # Preprocess the input
    features = np.array([[tv, radio, newspaper]])
    features_scaled = scaler.transform(features)

    # Predict sales
    prediction = model.predict(features_scaled)
    print(prediction)
    # Return the result
    return render_template("index.html", prediction=f"Predicted Sales: {prediction[0]:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
