## Student Performance Prediction System


This project is designed to fetch student records from an external API, process the data, make predictions on their academic performance, and log the results back into the database. It utilizes machine learning models to predict the average academic score of a student based on their math, reading, and writing scores.

**Overview**

The system performs the following steps:

**Fetches the latest student record from the API.**

-Loads a pre-trained machine learning model and data scaler.

-Prepares the student data for prediction, handling missing values and ensuring the data is in the correct format.

-Makes a prediction based on the student’s information.

-Logs the prediction back into the API’s database.

**Requirements**

Before running this project, ensure you have the following installed:

-Python 3.x

-requests (for making API calls)

-pandas (for data manipulation)

-numpy (for numerical operations)

-joblib (for loading the machine learning model and scaler)

-scikit-learn (for model scaling)

**You can install these dependencies using pip:**
pip install requests pandas numpy joblib scikit-learn

## How It Works

**Step 1: Fetch the Latest Record from the API**

The system fetches the latest student record from the API at https://student-performance-api-p46u.onrender.com/students/. It retrieves all student records, identifies the latest record by the highest student ID, and returns this record for processing.

def fetch_latest_record():
    # Fetch the latest student record from the API
    ...
**Step 2: Load the Pre-trained Model and Scaler**

The system loads the pre-trained machine learning model and scaler using the joblib library. These files must be provided in the configuration file. The model is responsible for making the predictions, while the scaler standardizes the data before feeding it into the model.

def load_model_and_scaler(model_path, scaler_path):
    # Load model and scaler for prediction
    ...

**Step 3: Handle Missing Data and Prepare Input Data**

Data preparation involves several tasks:

**Handling Missing Values:**

Missing data is detected and handled by filling categorical columns with their most frequent value (mode).

**Feature Engineering:**

If necessary, we calculate additional features, such as the average score.

**Data Transformation:**

We ensure that column names match the format expected by the model and apply one-hot encoding to categorical variables.
Data Scaling: We scale the data to make sure it matches the format used during training.

def prepare_data(record, scaler):
    # Handle missing data, feature engineering, encoding, and scaling
    ...

**Step 4: Make Predictions**

Once the data is ready, the system makes a prediction using the pre-trained model. The model predicts the student’s average score based on the provided features.

**Step 5: Log the Prediction Back to the Database**

After the prediction is made, the result is logged back into the API’s database. The prediction is saved alongside the student's original data.

def log_prediction_to_db(prediction, student_id, student_data):
    # Log the predicted average score to the database
    ...

**Running the System**

-Ensure you have the pre-trained model (best_student_performance_model.pkl) and scaler (scaler.pkl) saved in the correct directory.

-Set the paths to these files in the main script.

-Run the script.

**python student_performance_prediction.py**

**This will:**

-Fetch the latest student record.

-Prepare the data and make a prediction.

-Log the prediction back to the API.

**Error Handling**

Throughout the process, the system handles various exceptions to ensure smooth execution:

**-Errors in fetching data from the API are caught and printed.**

**-Errors in loading the model or scaler are handled.**

**-Missing or invalid data is logged, and the program exits gracefully if critical data is missing.**

**-If the prediction cannot be made, the system will notify the user.**

## Contributing

We welcome contributions to improve the system. If you'd like to add new features or fix bugs, please feel free to fork the repository, make changes, and submit a pull request.
