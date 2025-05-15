import joblib
 
model = joblib.load("./model/rf_model.joblib")
result_target = joblib.load("./model/encoder_target.joblib")

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