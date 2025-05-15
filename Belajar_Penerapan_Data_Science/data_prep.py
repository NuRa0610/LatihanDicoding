import joblib
import numpy as np
import pandas as pd
import os

encoder_Application_mode = joblib.load('model/encoder_Application_mode.joblib')
encoder_Course = joblib.load('model/encoder_Course.joblib')
encoder_Daytime_evening_attendance = joblib.load('model/encoder_Daytime_evening_attendance.joblib')
encoder_Debtor = joblib.load('model/encoder_Debtor.joblib')
encoder_Displaced = joblib.load('model/encoder_Displaced.joblib')
encoder_Educational_special_needs = joblib.load('model/encoder_Educational_special_needs.joblib')
encoder_Fathers_occupation = joblib.load('model/encoder_Fathers_occupation.joblib')
encoder_Fathers_qualification = joblib.load('model/encoder_Fathers_qualification.joblib')
encoder_Gender = joblib.load('model/encoder_Gender.joblib')
encoder_International = joblib.load('model/encoder_International.joblib')
encoder_Marital_status = joblib.load('model/encoder_Marital_status.joblib')
encoder_Mothers_occupation = joblib.load('model/encoder_Mothers_occupation.joblib')
encoder_Mothers_qualification = joblib.load('model/encoder_Mothers_qualification.joblib')
encoder_Nacionality = joblib.load('model/encoder_Nacionality.joblib')
encoder_Previous_qualification = joblib.load('model/encoder_Previous_qualification.joblib')
encoder_Scholarship_holder = joblib.load('model/encoder_Scholarship_holder.joblib')
encoder_target = joblib.load('model/encoder_target.joblib')
encoder_Tuition_fees_up_to_date = joblib.load('model/encoder_Tuition_fees_up_to_date.joblib')
scaler_Admission_grade = joblib.load('model/scaler_Admission_grade.joblib')
scaler_Curricular_units_1st_sem_approved = joblib.load('model/scaler_Curricular_units_1st_sem_approved.joblib')
scaler_Curricular_units_1st_sem_enrolled = joblib.load('model/scaler_Curricular_units_1st_sem_enrolled.joblib')
scaler_Curricular_units_1st_sem_credited = joblib.load('model/scaler_Curricular_units_1st_sem_credited.joblib')
scaler_Curricular_units_1st_sem_evaluations = joblib.load('model/scaler_Curricular_units_1st_sem_evaluations.joblib')
scaler_Curricular_units_1st_sem_grade = joblib.load('model/scaler_Curricular_units_1st_sem_grade.joblib')
scaler_Curricular_units_1st_sem_without_evaluations = joblib.load('model/scaler_Curricular_units_1st_sem_without_evaluations.joblib')
scaler_Curricular_units_2nd_sem_approved = joblib.load('model/scaler_Curricular_units_2nd_sem_approved.joblib')
scaler_Curricular_units_2nd_sem_enrolled = joblib.load('model/scaler_Curricular_units_2nd_sem_enrolled.joblib')
scaler_Curricular_units_2nd_sem_credited = joblib.load('model/scaler_Curricular_units_2nd_sem_credited.joblib')
scaler_Curricular_units_2nd_sem_evaluations = joblib.load('model/scaler_Curricular_units_2nd_sem_evaluations.joblib')
scaler_Curricular_units_2nd_sem_grade = joblib.load('model/scaler_Curricular_units_2nd_sem_grade.joblib')
scaler_Curricular_units_2nd_sem_without_evaluations = joblib.load('model/scaler_Curricular_units_2nd_sem_without_evaluations.joblib')
scaler_GDP = joblib.load('model/scaler_GDP.joblib')
scaler_Inflation_rate = joblib.load('model/scaler_Inflation_rate.joblib')
scaler_Previous_qualification_grade = joblib.load('model/scaler_Previous_qualification_grade.joblib')
scaler_Unemployment_rate = joblib.load('model/scaler_Unemployment_rate.joblib')

def preprocess_data(data):
    """Preprocess the data for prediction

    Args:
        data (Pandas DataFrame): Dataframe that contain all the data

    Returns:
        Pandas DataFrame: Preprocessed data
    """
    # Drop columns that are not needed
    #data = data.drop(columns=[''])

    # Encode categorical variables
    data['Application_mode'] = encoder_Application_mode.transform(data['Application_mode'])
    data['Course'] = encoder_Course.transform(data['Course'])
    data['Fathers_occupation'] = encoder_Fathers_occupation.transform(data['Fathers_occupation'])
    data['Fathers_qualification'] = encoder_Fathers_qualification.transform(data['Fathers_qualification'])
    data['Marital_status'] = encoder_Marital_status.transform(data['Marital_status'])
    data['Mothers_occupation'] = encoder_Mothers_occupation.transform(data['Mothers_occupation'])
    data['Mothers_qualification'] = encoder_Mothers_qualification.transform(data['Mothers_qualification'])
    data['Nacionality'] = encoder_Nacionality.transform(data['Nacionality'])
    data['Previous_qualification'] = encoder_Previous_qualification.transform(data['Previous_qualification'])
    
    #Onehot encoding
    encoded_daytime_evening_attendance = encoder_Daytime_evening_attendance.transform(data[['Daytime_evening_attendance']])
    encoded_daytime_evening_attendance = pd.DataFrame(
        encoded_daytime_evening_attendance,
        columns=encoder_Daytime_evening_attendance.get_feature_names_out(['Daytime_evening_attendance'])
    )
    data = pd.concat([data.drop(columns=['Daytime_evening_attendance']), encoded_daytime_evening_attendance], axis=1)
    
    encoded_displaced = encoder_Displaced.transform(data[['Displaced']])
    encoded_displaced = pd.DataFrame(
        encoded_displaced,
        columns=encoder_Displaced.get_feature_names_out(['Displaced'])
    )
    data = pd.concat([data.drop(columns=['Displaced']), encoded_displaced], axis=1)
    #data['Displaced'] = encoder_Displaced.transform(data[['Displaced']])
    
    encoded_educational_special_needs = encoder_Educational_special_needs.transform(data[['Educational_special_needs']])
    encoded_educational_special_needs = pd.DataFrame(
        encoded_educational_special_needs,
        columns=encoder_Educational_special_needs.get_feature_names_out(['Educational_special_needs'])
    )
    data = pd.concat([data.drop(columns=['Educational_special_needs']), encoded_educational_special_needs], axis=1)
    #data['Educational_special_needs'] = encoder_Educational_special_needs.transform(data[['Educational_special_needs']])
    
    encoded_debtor = encoder_Debtor.transform(data[['Debtor']])
    encoded_debtor = pd.DataFrame(
        encoded_debtor,
        columns=encoder_Debtor.get_feature_names_out(['Debtor'])
    )
    data = pd.concat([data.drop(columns=['Debtor']), encoded_debtor], axis=1)

    encoded_tuition_fees_up_to_date = encoder_Tuition_fees_up_to_date.transform(data[['Tuition_fees_up_to_date']])
    encoded_tuition_fees_up_to_date = pd.DataFrame(
        encoded_tuition_fees_up_to_date,
        columns=encoder_Tuition_fees_up_to_date.get_feature_names_out(['Tuition_fees_up_to_date'])
    )
    data = pd.concat([data.drop(columns=['Tuition_fees_up_to_date']), encoded_tuition_fees_up_to_date], axis=1)
    #data['Tuition_fees_up_to_date'] = encoder_Tuition_fees_up_to_date.transform(data[['Tuition_fees_up_to_date']])

    encoded_gender = encoder_Gender.transform(data[['Gender']])
    encoded_gender = pd.DataFrame(
        encoded_gender,
        columns=encoder_Gender.get_feature_names_out(['Gender'])
    )
    data = pd.concat([data.drop(columns=['Gender']), encoded_gender], axis=1)
    #data['Gender'] = encoder_Gender.transform(data[['Gender']])
    
    encoded_scholarship_holder = encoder_Scholarship_holder.transform(data[['Scholarship_holder']])
    encoded_scholarship_holder = pd.DataFrame(
        encoded_scholarship_holder,
        columns=encoder_Scholarship_holder.get_feature_names_out(['Scholarship_holder'])
    )
    data = pd.concat([data.drop(columns=['Scholarship_holder']), encoded_scholarship_holder], axis=1)
    #data['Scholarship_holder'] = encoder_Scholarship_holder.transform(data[['Scholarship_holder']])
        
    encoded_international = encoder_International.transform(data[['International']])
    encoded_international = pd.DataFrame(
        encoded_international,
        columns=encoder_International.get_feature_names_out(['International'])
    )
    data = pd.concat([data.drop(columns=['International']), encoded_international], axis=1)
    #data['International'] = encoder_International.transform(data[['International']])
    
    # Scale numerical variables
    data['Admission_grade'] = scaler_Admission_grade.transform(data[['Admission_grade']])
    data['Curricular_units_1st_sem_approved'] = scaler_Curricular_units_1st_sem_approved.transform(data[['Curricular_units_1st_sem_approved']])
    data['Curricular_units_1st_sem_enrolled'] = scaler_Curricular_units_1st_sem_enrolled.transform(data[['Curricular_units_1st_sem_enrolled']])
    data['Curricular_units_1st_sem_credited'] = scaler_Curricular_units_1st_sem_credited.transform(data[['Curricular_units_1st_sem_credited']])
    data['Curricular_units_1st_sem_evaluations'] = scaler_Curricular_units_1st_sem_evaluations.transform(data[['Curricular_units_1st_sem_evaluations']])
    data['Curricular_units_1st_sem_grade'] = scaler_Curricular_units_1st_sem_grade.transform(data[['Curricular_units_1st_sem_grade']])
    data['Curricular_units_1st_sem_without_evaluations'] = scaler_Curricular_units_1st_sem_without_evaluations.transform(data[['Curricular_units_1st_sem_without_evaluations']])
    data['Curricular_units_2nd_sem_approved'] = scaler_Curricular_units_2nd_sem_approved.transform(data[['Curricular_units_2nd_sem_approved']])
    data['Curricular_units_2nd_sem_enrolled'] = scaler_Curricular_units_2nd_sem_enrolled.transform(data[['Curricular_units_2nd_sem_enrolled']])
    data['Curricular_units_2nd_sem_credited'] = scaler_Curricular_units_2nd_sem_credited.transform(data[['Curricular_units_2nd_sem_credited']])
    data['Curricular_units_2nd_sem_evaluations'] = scaler_Curricular_units_2nd_sem_evaluations.transform(data[['Curricular_units_2nd_sem_evaluations']])
    data['Curricular_units_2nd_sem_grade'] = scaler_Curricular_units_2nd_sem_grade.transform(data[['Curricular_units_2nd_sem_grade']])
    data['Curricular_units_2nd_sem_without_evaluations'] = scaler_Curricular_units_2nd_sem_without_evaluations.transform(data[['Curricular_units_2nd_sem_without_evaluations']])
    data['GDP'] = scaler_GDP.transform(data[['GDP']])
    data['Inflation_rate'] = scaler_Inflation_rate.transform(data[['Inflation_rate']])
    data['Previous_qualification_grade'] = scaler_Previous_qualification_grade.transform(data[['Previous_qualification_grade']])
    data['Unemployment_rate'] = scaler_Unemployment_rate.transform(data[['Unemployment_rate']])

    # Urutan kolom yang diinginkan
    desired_column_order = [
        'Marital_status', 'Application_mode', 'Application_order', 'Course',
        'Previous_qualification', 'Previous_qualification_grade', 'Nacionality',
        'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation',
        'Fathers_occupation', 'Admission_grade', 'Age_at_enrollment',
        'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
        'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
        'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
        'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
        'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
        'Unemployment_rate', 'Inflation_rate', 'GDP',
        'Daytime_evening_attendance_0', 'Daytime_evening_attendance_1',
        'Displaced_0', 'Displaced_1', 'Educational_special_needs_0',
        'Educational_special_needs_1', 'Debtor_0', 'Debtor_1',
        'Tuition_fees_up_to_date_0', 'Tuition_fees_up_to_date_1',
        'Gender_0', 'Gender_1', 'Scholarship_holder_0', 'Scholarship_holder_1',
        'International_0', 'International_1'
    ]

    # Reorder columns
    data = data.reindex(columns=desired_column_order)

    return data