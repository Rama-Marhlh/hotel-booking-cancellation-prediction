from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle
import numpy as np
from datetime import datetime
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the scaler and the model
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('rf_model.pkl', 'rb'))

meal_plan_map = {
    'Meal Plan 1': 1,
    'Meal Plan 2': 2,
    'Not Selected': 0,
}

room_type_map = {
    'Room_Type 1': 1,
    'Room_Type 2': 2,
    'Room_Type 3': 3,
    'Room_Type 4': 4,
    'Room_Type 5': 5,
    'Room_Type 6': 6,
    'Room_Type 7': 7
}

market_segment_map = {
    'Aviation': 1,
    'Complementary': 2,
    'Corporate': 3,
    'Offline': 4,
    'Online': 5
}

# Define expected columns based on preprocessing
expected_columns = [
    'number of adults', 'number of children', 'number of weekend nights', 'number of week nights',
    'car parking space', 'lead time', 'repeated', 'P-C', 'P-not-C', 'average price',
    'special requests', 'day', 'month', 'year', 'Meal Plan 1', 'Meal Plan 2', 'Not Selected',
    'Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7',
    'Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_prelight_response()

    try:
        data = request.get_json()

        # Convert categorical variables to numerical values
        meal_plan = meal_plan_map.get(data['mealPlan'], 0)
        room_type = room_type_map.get(data['roomType'], 0)
        market_segment = market_segment_map.get(data['marketSegment'], 0)

        # Prepare the features for prediction
        features = {
            'number of adults': int(data['numAdults']),
            'number of children': int(data['numChildren']),
            'number of weekend nights': int(data['numWeekendNights']),
            'number of week nights': int(data['numWeekNights']),
            'car parking space': int(data['carParkingSpace']),
            'lead time': int(data['leadTime']),
            'repeated': int(data['repeated']),
            'P-C': int(data['pC']),
            'P-not-C': int(data['pNotC']),
            'average price': float(data['averagePrice']),  # Make sure this matches what model expects
            'special requests': int(data['specialRequests']),
            'day': int(datetime.fromisoformat(data['reservationDate']).day),
            'month': int(datetime.fromisoformat(data['reservationDate']).month),
            'year': int(datetime.fromisoformat(data['reservationDate']).year),
            'Meal Plan 1': 1 if data['mealPlan'] == 'Meal Plan 1' else 0,
            'Meal Plan 2': 1 if data['mealPlan'] == 'Meal Plan 2' else 0,
            'Not Selected': 1 if data['mealPlan'] == 'Not Selected' else 0,
            'Room_Type 1': 1 if data['roomType'] == 'Room_Type 1' else 0,
            'Room_Type 2': 1 if data['roomType'] == 'Room_Type 2' else 0,
            'Room_Type 3': 1 if data['roomType'] == 'Room_Type 3' else 0,
            'Room_Type 4': 1 if data['roomType'] == 'Room_Type 4' else 0,
            'Room_Type 5': 1 if data['roomType'] == 'Room_Type 5' else 0,
            'Room_Type 6': 1 if data['roomType'] == 'Room_Type 6' else 0,
            'Room_Type 7': 1 if data['roomType'] == 'Room_Type 7' else 0,
            'Aviation': 1 if data['marketSegment'] == 'Aviation' else 0,
            'Complementary': 1 if data['marketSegment'] == 'Complementary' else 0,
            'Corporate': 1 if data['marketSegment'] == 'Corporate' else 0,
            'Offline': 1 if data['marketSegment'] == 'Offline' else 0,
            'Online': 1 if data['marketSegment'] == 'Online' else 0
        }

        # Create a DataFrame from features
        features_df = pd.DataFrame([features])

        # Reorder columns to match expected order
        features_df = features_df[expected_columns]

        # Scale the features
        features_scaled = scaler.transform(features_df)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_result = "Canceled" if prediction == 1 else "Not Canceled"

        return jsonify({"prediction": prediction_result})

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 400


def _build_cors_prelight_response():
    response = jsonify()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response, 200


if __name__ == "__main__":
    app.run(debug=True)


# ### Detailed Explanation of the Prediction Process

# #### 1. Form Submission
# - **User Action:** When a user fills out the form and clicks the "Predict" button, a JavaScript function is triggered to handle the submission.

# #### 2. JavaScript Function - `predict()`
# - **Function Responsibility:** This function collects data from the form, prepares it for the server, and handles the server's response.

# #### 3. Collecting Form Data
# - **Selecting Form Element:** The function identifies the form element in the web page.
# - **Creating Form Data Object:** It automatically gathers all the input fields and their values into a structured format.

# #### 4. Converting Form Data to JSON
# - **Converting to Plain Object:** The gathered data is converted into a simple JavaScript object.
# - **Serializing to JSON:** This object is then turned into a JSON string, a text-based format that is easy for the server to understand.

# #### 5. Sending Data to the Server
# - **Using Fetch API:** A network request is made to the server's endpoint, indicating the type of request (POST), specifying that the data format is JSON, and including the serialized data.

# #### 6. Handling the Server Response
# - **Parsing Response as JSON:** The response from the server is read and converted back into a JavaScript object.

# #### 7. Displaying the Result
# - **Handling Result:** The result from the server is displayed to the user. If it's a prediction, it is shown in one part of the web page; if there's an error, it is shown in another part.

# ### Summary of the Flow

# 1. **User Action:** The user fills out the form and clicks "Predict".
# 2. **Form Data Collection:** The function gathers all the form data.
# 3. **Data Conversion:** The form data is converted to JSON format.
# 4. **Sending Request:** The data is sent to the server using Fetch API.
# 5. **Server Processing:** The server processes the request and sends back a prediction.
# 6. **Displaying Results:** The prediction result or error message is displayed to the user.

# ### Why Convert to JSON?
# - **Language-Independent:** JSON can be used by any programming language.
# - **Structured:** It maintains the structure of the data.
# - **Lightweight:** JSON is compact and efficient for communication.

# ### What is the Fetch API?
# - **Modern Interface:** It allows making network requests.
# - **Promises-Based:** It makes handling asynchronous operations easier.
# - **Simpler Syntax:** It is more intuitive to use.
# - **Various Methods:** Supports different types of HTTP requests.
# - **Streaming:** Can handle large data sets efficiently.

# ### Local Development and Server Interaction
# - **Local Server:** You run your Flask application locally, which listens for requests.
# - **Client-Side Code:** Your web page uses Fetch API to communicate with this local server.
# - **Network Request:** Fetch API sends a request to the server's endpoint.
# - **Server Response:** The server processes the request and sends back a response, which is handled by the Fetch API in your JavaScript code.

# ### Making the Form Accessible to Anyone

# 1. **Prepare Your Flask Application:** Ensure it works correctly locally.
# 2. **Choose a Hosting Provider:** Options include Heroku, AWS, Google Cloud, etc.
# 3. **Deploy on Heroku:** Example steps for deploying on Heroku:
#    - Install Heroku CLI and log in.
#    - Create necessary files for deployment.
#    - Initialize a Git repository.
#    - Create Heroku app and deploy.
#    - Open and share the provided URL.

# By following these steps, your Flask application, including the form, will be accessible to anyone with the link provided by the hosting provider.