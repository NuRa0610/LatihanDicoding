import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Load model and encoder
model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.joblib'))
result_target = joblib.load(os.path.join(MODEL_DIR, 'encoder_target.joblib'))

def prediction(data):
    """Making prediction

    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data

    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    result = model.predict(data)
    #print("Raw prediction from model:", result)
    final_result = result_target.inverse_transform(result)[0]
    return final_result

def map_prediction_to_binary(prediction):
    """Map model prediction to binary classes (Dropout or Tidak Dropout)."""
    if prediction == "Dropout":  # Dropout
        return "Dropout"
    else:  # Enrolled or Graduate
        return "Tidak Dropout"