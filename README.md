# hotel-booking-cancellation-prediction


---

**Project Title**: Booking Prediction with Machine Learning

**Description**:
This project focuses on predicting hotel booking cancellations using machine learning techniques. It includes data preprocessing, model training, and deploying a prediction system via a web application.

**Key Features**:
1. **Dataset**: The dataset comprises various features such as lead time, average price, special requests, booking dates, and customer demographics.
   Kaggle : https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction
   
3. **Preprocessing**: Data preprocessing involved handling missing values, feature engineering, and encoding categorical variables using one-hot encoding.

4. **Model Training**: Used PyCaret, a low-code machine learning library in Python, to train multiple classification models such as Random Forest, XGBoost, and Extra Trees. Model performance was evaluated using cross-validation.

5. **Feature Selection**: Employed feature importance techniques to select relevant features for model training, optimizing prediction accuracy.

6. **Model Blending**: Utilized model blending via PyCaret's `blend_models` function to combine predictions from multiple models (Random Forest, XGBoost, and Extra Trees) to enhance predictive performance.

7. **Deployment**: Developed a Flask web application for deploying the final blended model. The application allows users to input booking details and receive real-time predictions on whether a booking is likely to be canceled.

**Technologies Used**:
- Python
- PyCaret
- Flask
- HTML/CSS
- JavaScript (Fetch API)

**Repository Structure**:
- **app.py**: Flask application for prediction endpoint and web interface.
- **templates/index.html**: HTML template for the web interface.
- **static/**: Directory for static files (e.g., CSS stylesheets, images).
- **data/**: Directory containing the dataset used for training and testing.
- **models/**: Directory to store trained machine learning models (if applicable).
- **requirements.txt**: List of Python dependencies for easy installation.


---
![image](https://github.com/Rama-Marhlh/hotel-booking-cancellation-prediction/assets/90220172/b2c624cd-063f-493a-b1be-5f9b401becc2)



![image](https://github.com/Rama-Marhlh/hotel-booking-cancellation-prediction/assets/90220172/52f9a524-8a6f-427d-94f6-70811dcf0028)
