import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Step 1: Fetch the Latest Entry from the API
def fetch_latest_record():
    api_url_all_records = "https://student-performance-api-p46u.onrender.com/students/"
    try:
        response = requests.get(api_url_all_records)
        response.raise_for_status()  # Raise an error for bad responses
        all_records = response.json()
        
        # Extract the latest record based on 'id' or another unique identifier
        if isinstance(all_records, list) and len(all_records) > 0:
            latest_record = max(all_records, key=lambda x: x.get("student_id", 0))
            print("Fetched Latest Record:", latest_record)
            return latest_record
        else:
            print("No records found in the database.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

# Step 2: Load the Pre-trained Model and Scaler
def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and Scaler Loaded Successfully")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

# Step 3: Handle Missing Data and Prepare Input Data for Prediction
def prepare_data(record, scaler):
    try:
        # Create a DataFrame from the record
        df = pd.DataFrame([record])

        # Step 3.1: Handle Missing Data
        if df.isnull().sum().sum() > 0:
            print("Missing data detected. Handling missing values.")
            # # Fill missing values with appropriate defaults (e.g., median, mode, or a constant value)
            # df.fillna(df.mean(), inplace=True)  # For numerical columns, use the mean for missing data
            # # Alternatively, handle categorical features using the mode (most frequent category)
            categorical_cols = ["gender", "race_ethnicity", "lunch", "education_level_id", "test_preparation_id"]
            for col in categorical_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Step 3.2: Feature Engineering
        if all(col in df.columns for col in ["math_score", "reading_score", "writing_score"]):
            # Calculate "average_score" but do not include it in the input data for prediction
            df["average_score"] = df[["math_score", "reading_score", "writing_score"]].mean(axis=1)
            df.drop(["math_score", "reading_score", "writing_score"], axis=1, inplace=True)
        else:
            print("Missing required score columns in input data.")
            return None

        # Step 3.3: Rename API column names to match model training names
        column_mapping = {
            "education_level_id": "parental level of education",
            "test_preparation_id": "test preparation course",
            "race_ethnicity": "race/ethnicity",
            "lunch": "lunch",
            "gender": "gender"
        }
        df.rename(columns=column_mapping, inplace=True)

        # Convert numerical categories to categorical labels
        education_mapping = {
            1: "some high school", 2: "high school", 3: "some college",
            4: "associate's degree", 5: "bachelor's degree", 6: "master's degree"
        }
        test_prep_mapping = {1: "completed", 2: "none"}

        df["parental level of education"] = df["parental level of education"].map(education_mapping)
        df["test preparation course"] = df["test preparation course"].map(test_prep_mapping)

        # Step 3.4: One-Hot Encoding
        categorical_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        # Define the expected feature names (in the correct order)
        expected_columns = [
            "gender_male",
            "race/ethnicity_group B",
            "race/ethnicity_group C",
            "race/ethnicity_group D",
            "race/ethnicity_group E",
            "parental level of education_bachelor's degree",
            "parental level of education_high school",
            "parental level of education_master's degree",
            "parental level of education_some college",
            "parental level of education_some high school",
            "lunch_standard",
            "test preparation course_none"
        ]

        # Align DataFrame with expected model features
        df = df.reindex(columns=expected_columns, fill_value=0)

        # Step 3.5: Scaling the Data
        df_scaled = scaler.transform(df)
        print("Prepared and Scaled Input Data:", df_scaled)
        return df_scaled
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return None

# Step 4: Make Predictions
def make_prediction(model, data):
    try:
        if data is None:
            print("Data preparation failed. Skipping prediction.")
            return None
        prediction = model.predict(data)
        print("Prediction Made:", prediction)
        return prediction[0]  # Return the first prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

# Step 5: Log the Prediction Result back to the Database
def log_prediction_to_db(prediction, student_id, student_data):
    api_url_update_record = f"https://student-performance-api-p46u.onrender.com/students/{student_id}"
    payload = {
        "predicted_average_score": prediction,
        # Include the missing fields along with the prediction
        "gender": student_data.get("gender"),
        "race_ethnicity": student_data.get("race_ethnicity"),
        "lunch": student_data.get("lunch"),
        "education_level_id": student_data.get("education_level_id"),
        "test_preparation_id": student_data.get("test_preparation_id"),
        "math_score": student_data.get("math_score"),
        "reading_score": student_data.get("reading_score"),
        "writing_score": student_data.get("writing_score")
    }
    try:
        response = requests.put(api_url_update_record, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        print("Prediction logged to the database successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error logging prediction to database: {e}")

# Main Function
if __name__ == "__main__":
    # Paths and Configuration
    MODEL_PATH = "/Users/mac/Documents/TECHIE/Models/best_student_performance_model.pkl"
    SCALER_PATH = "/Users/mac/Documents/TECHIE/Models/scaler.pkl"

    # Fetch the latest record
    latest_record = fetch_latest_record()
    if not latest_record:
        print("Exiting due to failure in fetching the latest record.")
        exit(1)

    # Load the pre-trained model and scaler
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    if not model or not scaler:
        print("Exiting due to failure in loading the model or scaler.")
        exit(1)

    # Prepare the input data
    input_data = prepare_data(latest_record, scaler)
    if input_data is None:
        print("Exiting due to failure in preparing input data.")
        exit(1)

    # Make a prediction
    prediction = make_prediction(model, input_data)
    if prediction is None:
        print("Exiting due to failure in making a prediction.")
        exit(1)

    # Print the final prediction
    print(f"Final Prediction for Average Score: {prediction}")

    # Step 5: Log the result back into the database
    student_id = latest_record.get("student_id", None)
    if student_id:
        log_prediction_to_db(prediction, student_id, latest_record)
    else:
        print("No student ID found. Unable to log the prediction to the database.")