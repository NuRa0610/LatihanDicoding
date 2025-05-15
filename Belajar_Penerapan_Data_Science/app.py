import streamlit as st
import pandas as pd
import joblib
from data_prep import preprocess_data
from data_prep import (
    encoder_Application_mode,
    encoder_Course,
    encoder_Daytime_evening_attendance,
    encoder_Debtor,
    encoder_Displaced,
    encoder_Educational_special_needs,
    encoder_Fathers_occupation,
    encoder_Fathers_qualification,
    encoder_Gender,
    encoder_International,
    encoder_Marital_status,
    encoder_Mothers_occupation,
    encoder_Mothers_qualification,
    encoder_Nacionality,
    encoder_Previous_qualification,
    encoder_Scholarship_holder,
    encoder_Tuition_fees_up_to_date,
    encoder_target
)
from data_prep import (
    scaler_Admission_grade,
    scaler_Curricular_units_1st_sem_approved,
    scaler_Curricular_units_1st_sem_enrolled,
    scaler_Curricular_units_1st_sem_credited,
    scaler_Curricular_units_1st_sem_evaluations,
    scaler_Curricular_units_1st_sem_grade,
    scaler_Curricular_units_1st_sem_without_evaluations,
    scaler_Curricular_units_2nd_sem_approved,
    scaler_Curricular_units_2nd_sem_enrolled,
    scaler_Curricular_units_2nd_sem_credited,
    scaler_Curricular_units_2nd_sem_evaluations,
    scaler_Curricular_units_2nd_sem_grade,
    scaler_Curricular_units_2nd_sem_without_evaluations,
    scaler_GDP,
    scaler_Inflation_rate,
    scaler_Previous_qualification_grade,
    scaler_Unemployment_rate
)
from predict import prediction, map_prediction_to_binary

col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=130)  # Ikon sekolah
with col2:
    st.header('Jaya Institut (Prototype)')

data = pd.DataFrame()
 
col1, col2, col3 = st.columns([1,2,1])
 
with col1:
    Gender = st.selectbox(
        label='Gender',
        options=encoder_Gender.categories_[0],
        format_func=lambda x: {
            '0': 'Woman',
            '1': 'Gentleman'
        }.get(str(x), 'Unknown'),
        index=1
    )
    data["Gender"] = [Gender]
 
with col2:
    Marital_status = st.selectbox(
        label='Marital Status',
        options=encoder_Marital_status.classes_,
        format_func=lambda x: {
            '1': 'Single',
            '2': 'Married',
            '3': 'Widower',
            '4': 'Divorced',
            '5': 'Facto Union',
            '6': 'Legally Separated'
        }.get(str(x), 'Unknown'),
        index=0
    )
    data["Marital_status"] = [Marital_status]

with col3:
       
    Age_at_enrollment= int(st.number_input(
        label='Age', value=20, min_value=20, max_value=60,
        help="Age at enrollment (20 - 60 years old)"))
    data["Age_at_enrollment"] = Age_at_enrollment
 
col1, col2, col3 = st.columns([2,1,1])
 
with col1:

    Application_mode = st.selectbox(
        label='Application_mode',
        options=encoder_Application_mode.classes_,
        format_func=lambda x: {
            '1': '1st phase - general contingent',
            '2': 'Ordinance No. 612/93',
            '5': '1st phase - special contingent (Azores Island)',
            '7': 'Holders of other higher courses',
            '10': 'Ordinance No. 854-B/99',
            '15': 'International student (bachelor)',
            '16': '1st phase - special contingent (Madeira Island)',
            '17': '2nd phase - general contingent',
            '18': '3rd phase - general contingent',
            '26': 'Ordinance No. 533-A/99, item b2 (Different Plan)',
            '27': 'Ordinance No. 533-A/99, item b3 (Other Institution)',
            '39': 'Over 23 years old',
            '42': 'Transfer',
            '43': 'Change of course',
            '44': 'Technological specialization diploma holders',
            '51': 'Change of institution/course',
            '53': 'Short cycle diploma holders',
            '57': 'Change of institution/course (International)'
        }.get(str(x), 'Unknown'),  # Gunakan str(x) untuk pencocokan
        help="Select the application mode."
    )
    data["Application_mode"] = Application_mode

with col2:
    Admission_grade = int(st.number_input(
        label='Admission grade', value=150,
        help="Admission grade (0 - 200)",
        min_value=0, max_value=200,))
    data["Admission_grade"] = Admission_grade

with col3:
    Application_order = int(st.number_input(
        label='Application order', value=1, min_value=1, max_value=9,
        help="Application order (first '1' - last '9')"))
    data["Application_order"] = Application_order
 
col1, col2, col3= st.columns([2, 1, 1])
 
with col1:
    Previous_qualification = st.selectbox(
        label='Previous Qualification',
        options=encoder_Previous_qualification.classes_,
        format_func=lambda x: {
            '1': 'Secondary education',
            '2': "Higher education - bachelor's degree",
            '3': 'Higher education - degree',
            '4': "Higher education - master's",
            '5': 'Higher education - doctorate',
            '6': 'Frequency of higher education',
            '9': '12th year of schooling - not completed',
            '10': '11th year of schooling - not completed',
            '12': 'Other - 11th year of schooling',
            '14': '10th year of schooling',
            '15': '10th year of schooling - not completed',
            '19': 'Basic education 3rd cycle (9th/10th/11th year) or equiv.',
            '38': 'Basic education 2nd cycle (6th/7th/8th year) or equiv.',
            '39': 'Technological specialization course',
            '40': 'Higher education - degree (1st cycle)',
            '42': 'Professional higher technical course',
            '43': 'Higher education - master (2nd cycle)'
        }.get(x, 'Unknown'),
        help="Select the previous qualification level."
    )
    data["Previous_qualification"] = Previous_qualification
 
with col2:
    Previous_qualification_grade = float(st.number_input(
        label='Previous grade', value=3.0,
        help="Previous qualification grade (0.0 - 4.0)",
        min_value=0.0, max_value=4.0,))
    data["Previous_qualification_grade"] = Previous_qualification_grade
 
with col3:
    Nacionality = st.selectbox(
        label='Nacionality',
        options=encoder_Nacionality.classes_,
        format_func=lambda x: {
            '1': 'Portuguese',
            '2': 'German',
            '6': 'Spanish',
            '11': 'Italian',
            '13': 'Dutch',
            '14': 'English',
            '17': 'Lithuanian',
            '21': 'Angolan',
            '22': 'Cape Verdean',
            '24': 'Guinean',
            '25': 'Mozambican',
            '26': 'Santomean',
            '32': 'Turkish',
            '41': 'Brazilian',
            '62': 'Romanian',
            '100': 'Moldova (Republic of)',
            '101': 'Mexican',
            '103': 'Ukrainian',
            '105': 'Russian',
            '108': 'Cuban',
            '109': 'Colombian'
        }.get(str(x), 'Unknown'),
        help="Select your nationality."
    )
    data["Nacionality"] = Nacionality
 
col1, col2 = st.columns(2)
 
with col1:
    Course = st.selectbox(
        label='Course',
        options=encoder_Course.classes_,
        format_func=lambda x: {
            '33': 'Biofuel Production Technologies',
            '171': 'Animation and Multimedia Design',
            '8014': 'Social Service (evening attendance)',
            '9003': 'Agronomy',
            '9070': 'Communication Design',
            '9085': 'Veterinary Nursing',
            '9119': 'Informatics Engineering',
            '9130': 'Equinculture',
            '9147': 'Management',
            '9238': 'Social Service',
            '9254': 'Tourism',
            '9500': 'Nursing',
            '9556': 'Oral Hygiene',
            '9670': 'Advertising and Marketing Management',
            '9773': 'Journalism and Communication',
            '9853': 'Basic Education',
            '9991': 'Management (evening attendance)'
        }.get(str(x), 'Unknown'),
        help="Select the course you are enrolled in."
    )
    data["Course"] = Course
 
with col2:
    Daytime_evening_attendance = st.selectbox(
        label='Daytime_evening_attendance',
        options=encoder_Daytime_evening_attendance.categories_[0],
        format_func=lambda x: {
            '0': 'Evening',
            '1': 'Daytime'
        }.get(str(x), 'Unknown'),
        help="Select the attendance type (Daytime or Evening)."
    )
    data["Daytime_evening_attendance"] = Daytime_evening_attendance

col1, col2, col3 = st.columns(3)

with col1:
    Educational_special_needs = st.selectbox(
        label='Educational_special_needs',
        options=encoder_Educational_special_needs.categories_[0],
        format_func=lambda x: {
            '0': 'No',
            '1': 'Yes'
        }.get(str(x), 'Unknown'),
        help="Select if the student has educational special needs."
    )
    data["Educational_special_needs"] = Educational_special_needs

with col2:
    Displaced = st.selectbox(
        label='Displaced',
        options=encoder_Displaced.categories_[0],
        format_func=lambda x: {
            '0': 'No',
            '1': 'Yes'
        }.get(str(x), 'Unknown'),
        help="Select if the student is displaced."
    )
    data["Displaced"] = Displaced

with col3:
    International = st.selectbox(
        label='International',
        options=encoder_International.categories_[0],
        format_func=lambda x: {
            '0': 'No',
            '1': 'Yes'
        }.get(str(x), 'Unknown'),
        help="Select if the student is international."
    )
    data["International"] = International

col1, col2, col3 = st.columns(3)

with col1:
    Debtor = st.selectbox(
        label='Debtor',
        options=encoder_Debtor.categories_[0],
        format_func=lambda x: {
            '0': 'No',
            '1': 'Yes'
        }.get(str(x), 'Unknown'),
        help="Select if the student is a debtor."
    )
    data["Debtor"] = Debtor

with col2:
    Scholarship_holder = st.selectbox(
        label='Scholarship_holder',
        options=encoder_Scholarship_holder.categories_[0],
        format_func=lambda x: {
            '0': 'No',
            '1': 'Yes'
        }.get(str(x), 'Unknown'),
        help="Select if the student is a scholarship holder."
    )
    data["Scholarship_holder"] = Scholarship_holder

with col3:
    Tuition_fees_up_to_date = st.selectbox(
        label='Tuition_fees_up_to_date',
        options=encoder_Tuition_fees_up_to_date.categories_[0],
        format_func=lambda x: {
            '0': 'No',
            '1': 'Yes'
        }.get(str(x), 'Unknown'),
        help="Select if the tuition fees are up to date."
    )
    data["Tuition_fees_up_to_date"] = Tuition_fees_up_to_date

col1, col2 = st.columns(2)

with col1:
    Mothers_occupation = st.selectbox(
        label='Mothers_occupation',
        options=encoder_Mothers_occupation.classes_,
        format_func=lambda x: {
            '0': 'Student',
            '1': 'Legislative Power Representatives, Directors, and Managers',
            '2': 'Specialists in Intellectual and Scientific Activities',
            '3': 'Intermediate Level Technicians and Professions',
            '4': 'Administrative staff',
            '5': 'Personal Services, Security Workers, and Sellers',
            '6': 'Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry',
            '7': 'Skilled Workers in Industry, Construction, and Craftsmen',
            '8': 'Installation and Machine Operators and Assembly Workers',
            '9': 'Unskilled Workers',
            '10': 'Armed Forces Professions',
            '90': 'Other Situation',
            '99': '(blank)',
            '122': 'Health professionals',
            '123': 'Teachers',
            '125': 'Specialists in ICT',
            '131': 'Intermediate Level Science and Engineering Technicians',
            '132': 'Intermediate Level Health Technicians',
            '134': 'Intermediate Level Legal, Social, and Cultural Technicians',
            '141': 'Office Workers and Secretaries',
            '143': 'Data, Accounting, and Financial Operators',
            '144': 'Other Administrative Support Staff',
            '151': 'Personal Service Workers',
            '152': 'Sellers',
            '153': 'Personal Care Workers',
            '171': 'Skilled Construction Workers',
            '173': 'Skilled Workers in Printing and Crafts',
            '175': 'Workers in Food Processing and Other Industries',
            '191': 'Cleaning Workers',
            '192': 'Unskilled Workers in Agriculture and Fisheries',
            '193': 'Unskilled Workers in Manufacturing and Transport',
            '194': 'Meal Preparation Assistants'
        }.get(str(x), 'Unknown'),
        help="Select the occupation of the mother."
    )
    data["Mothers_occupation"] = Mothers_occupation

with col2:
    Mothers_qualification = st.selectbox(
        label='Mothers_qualification',
        options=encoder_Mothers_qualification.classes_,
        format_func=lambda x: {
            '1': 'Secondary Education - 12th Year of Schooling or Eq.',
            '2': "Higher Education - Bachelor's Degree",
            '3': 'Higher Education - Degree',
            '4': "Higher Education - Master's",
            '5': 'Higher Education - Doctorate',
            '6': 'Frequency of Higher Education',
            '9': '12th Year of Schooling - Not Completed',
            '10': '11th Year of Schooling - Not Completed',
            '11': '7th Year (Old)',
            '12': 'Other - 11th Year of Schooling',
            '14': '10th Year of Schooling',
            '18': 'General Commerce Course',
            '19': 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
            '22': 'Technical-Professional Course',
            '26': '7th Year of Schooling',
            '27': '2nd Cycle of the General High School Course',
            '29': '9th Year of Schooling - Not Completed',
            '30': '8th Year of Schooling',
            '34': 'Unknown',
            '35': "Can't Read or Write",
            '36': 'Can Read Without Having a 4th Year of Schooling',
            '37': 'Basic Education 1st Cycle (4th/5th Year) or Equiv.',
            '38': 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
            '39': 'Technological Specialization Course',
            '40': 'Higher Education - Degree (1st Cycle)',
            '41': 'Specialized Higher Studies Course',
            '42': 'Professional Higher Technical Course',
            '43': 'Higher Education - Master (2nd Cycle)',
            '44': 'Higher Education - Doctorate (3rd Cycle)'
        }.get(str(x), 'Unknown'),
        help="Select the qualification of the mother."
    )
    data["Mothers_qualification"] = Mothers_qualification

col1, col2 = st.columns(2)

with col1:
    Fathers_occupation = st.selectbox(
        label='Fathers_occupation',
        options=encoder_Fathers_occupation.classes_,
        format_func=lambda x: {
            '0': 'Student',
            '1': 'Legislative Power Representatives, Directors, and Managers',
            '2': 'Specialists in Intellectual and Scientific Activities',
            '3': 'Intermediate Level Technicians and Professions',
            '4': 'Administrative staff',
            '5': 'Personal Services, Security Workers, and Sellers',
            '6': 'Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry',
            '7': 'Skilled Workers in Industry, Construction, and Craftsmen',
            '8': 'Installation and Machine Operators and Assembly Workers',
            '9': 'Unskilled Workers',
            '10': 'Armed Forces Professions',
            '90': 'Other Situation',
            '99': '(blank)',
            '101': 'Armed Forces Officers',
            '102': 'Armed Forces Sergeants',
            '103': 'Other Armed Forces Personnel',
            '112': 'Directors of Administrative and Commercial Services',
            '114': 'Hotel, Catering, Trade and Other Services Directors',
            '121': 'Specialists in Physical Sciences, Mathematics, Engineering, and Related Techniques',
            '122': 'Health Professionals',
            '123': 'Teachers',
            '124': 'Specialists in Finance, Accounting, Administrative Organization, Public and Commercial Relations',
            '131': 'Intermediate Level Science and Engineering Technicians',
            '132': 'Intermediate Level Health Technicians',
            '134': 'Intermediate Level Legal, Social, and Cultural Technicians',
            '135': 'Information and Communication Technology Technicians',
            '141': 'Office Workers and Secretaries',
            '143': 'Data, Accounting, and Financial Operators',
            '144': 'Other Administrative Support Staff',
            '151': 'Personal Service Workers',
            '152': 'Sellers',
            '153': 'Personal Care Workers',
            '154': 'Protection and Security Services Personnel',
            '161': 'Market-Oriented Farmers and Skilled Agricultural and Animal Production Workers',
            '163': 'Farmers, Livestock Keepers, Fishermen, Hunters, and Gatherers (Subsistence)',
            '171': 'Skilled Construction Workers',
            '172': 'Skilled Workers in Metallurgy, Metalworking, and Similar',
            '174': 'Skilled Workers in Electricity and Electronics',
            '175': 'Workers in Food Processing and Other Industries',
            '181': 'Fixed Plant and Machine Operators',
            '182': 'Assembly Workers',
            '183': 'Vehicle Drivers and Mobile Equipment Operators',
            '192': 'Unskilled Workers in Agriculture, Animal Production, Fisheries, and Forestry',
            '193': 'Unskilled Workers in Manufacturing and Transport',
            '194': 'Meal Preparation Assistants',
            '195': 'Street Vendors (Except Food) and Street Service Providers'
        }.get(str(x), 'Unknown'),
        help="Select the occupation of the father."
    )
    data["Fathers_occupation"] = Fathers_occupation

with col2:
    Fathers_qualification = st.selectbox(
        label='Fathers_qualification',
        options=encoder_Fathers_qualification.classes_,
        format_func=lambda x: {
            '1': 'Secondary Education - 12th Year of Schooling or Eq.',
            '2': "Higher Education - Bachelor's Degree",
            '3': 'Higher Education - Degree',
            '4': "Higher Education - Master's",
            '5': 'Higher Education - Doctorate',
            '6': 'Frequency of Higher Education',
            '9': '12th Year of Schooling - Not Completed',
            '10': '11th Year of Schooling - Not Completed',
            '11': '7th Year (Old)',
            '12': 'Other - 11th Year of Schooling',
            '13': '2nd year complementary high school course',
            '14': '10th Year of Schooling',
            '18': 'General Commerce Course',
            '19': 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
            '20': 'Complementary High School Course',
            '22': 'Technical-Professional Course',
            '25': 'Complementary High School Course - not concluded',
            '26': '7th Year of Schooling',
            '27': '2nd Cycle of the General High School Course',
            '29': '9th Year of Schooling - Not Completed',
            '30': '8th Year of Schooling',
            '31': 'General Course of Administration and Commerce',
            '33': 'Supplementary Accounting and Administration',
            '34': 'Unknown',
            '35': "Can't Read or Write",
            '36': 'Can Read Without Having a 4th Year of Schooling',
            '37': 'Basic Education 1st Cycle (4th/5th Year) or Equiv.',
            '38': 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
            '39': 'Technological Specialization Course',
            '40': 'Higher Education - Degree (1st Cycle)',
            '41': 'Specialized Higher Studies Course',
            '42': 'Professional Higher Technical Course',
            '43': 'Higher Education - Master (2nd Cycle)',
            '44': 'Higher Education - Doctorate (3rd Cycle)'
        }.get(str(x), 'Unknown'),
        help="Select the qualification of the father."
    )
    data["Fathers_qualification"] = Fathers_qualification

col1, col2, col3 = st.columns(3)
with col1:
    Curricular_units_1st_sem_approved = int(st.number_input(
        label='Curricular units 1st sem approved', value=0,
        help="Number of curricular units approved in the 1st semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_1st_sem_approved"] = Curricular_units_1st_sem_approved

with col2:
    Curricular_units_1st_sem_enrolled = int(st.number_input(
        label='Curricular units 1st sem enrolled', value=0,
        help="Number of curricular units enrolled in the 1st semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_1st_sem_enrolled"] = Curricular_units_1st_sem_enrolled

with col3:
    Curricular_units_1st_sem_credited = int(st.number_input(
        label='Curricular units 1st sem credited', value=0,
        help="Number of curricular units credited in the 1st semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_1st_sem_credited"] = Curricular_units_1st_sem_credited

col1, col2, col3 = st.columns(3)
with col1:
    Curricular_units_1st_sem_evaluations = int(st.number_input(
        label='Units 1st sem evaluations', value=0,
        help="Number of curricular units evaluations in the 1st semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_1st_sem_evaluations"] = Curricular_units_1st_sem_evaluations

with col2:
    Curricular_units_1st_sem_without_evaluations = int(st.number_input(
        label='Units 1st sem without evaluations', value=0,
        help="Number of curricular units without evaluations in the 1st semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_1st_sem_without_evaluations"] = Curricular_units_1st_sem_without_evaluations

with col3:
    Curricular_units_1st_sem_grade = float(st.number_input(
        label='Curricular units 1st sem grade', value=0.0,
        help="Curricular units grade in the 1st semester (0.0 - 20.0)",
        min_value=0.0, max_value=20.0,))
    data["Curricular_units_1st_sem_grade"] = Curricular_units_1st_sem_grade

col1, col2, col3 = st.columns(3)

with col1:
    Curricular_units_2nd_sem_approved = int(st.number_input(
        label='Curricular units 2nd sem approved', value=0,
        help="Number of curricular units approved in the 2nd semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_2nd_sem_approved"] = Curricular_units_2nd_sem_approved

with col2:
    Curricular_units_2nd_sem_enrolled = int(st.number_input(
        label='Curricular units 2nd sem enrolled', value=0,
        help="Number of curricular units enrolled in the 2nd semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_2nd_sem_enrolled"] = Curricular_units_2nd_sem_enrolled

with col3:
    Curricular_units_2nd_sem_credited = int(st.number_input(
        label='Curricular units 2nd sem credited', value=0,
        help="Number of curricular units credited in the 2nd semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_2nd_sem_credited"] = Curricular_units_2nd_sem_credited

col1, col2, col3 = st.columns(3)

with col1:
    Curricular_units_2nd_sem_evaluations = int(st.number_input(
        label='Units 2nd sem evaluations', value=0,
        help="Number of curricular units evaluations in the 2nd semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_2nd_sem_evaluations"] = Curricular_units_2nd_sem_evaluations

with col2:
    Curricular_units_2nd_sem_without_evaluations = int(st.number_input(
        label='Units 2nd sem without evaluations', value=0,
        help="Number of curricular units without evaluations in the 2nd semester (0 - 23)",
        min_value=0, max_value=23,))
    data["Curricular_units_2nd_sem_without_evaluations"] = Curricular_units_2nd_sem_without_evaluations

with col3:
    Curricular_units_2nd_sem_grade = float(st.number_input(
        label='Curricular units 2nd sem grade', value=0.0,
        help="Curricular units grade in the 2nd semester (0.0 - 20.0)",
        min_value=0.0, max_value=20.0,))
    data["Curricular_units_2nd_sem_grade"] = Curricular_units_2nd_sem_grade

col1, col2, col3 = st.columns(3)

with col1:
    GDP = float(st.number_input(
        label='GDP', value=0.0,
        help="GDP (0.0 - 100.0)",
        min_value=0.0, max_value=100.0,))
    data["GDP"] = GDP

with col2:
    Inflation_rate = float(st.number_input(
        label='Inflation rate', value=0.0,
        help="Inflation rate (0.0 - 100.0)",
        min_value=0.0, max_value=100.0,))
    data["Inflation_rate"] = Inflation_rate
    
with col3:
    Unemployment_rate = float(st.number_input(
        label='Unemployment rate', value=0.0,
        help="Unemployment rate (0.0 - 100.0)",
        min_value=0.0, max_value=100.0,))
    data["Unemployment_rate"] = Unemployment_rate

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

if st.button('Predict'):
    new_data = preprocess_data(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    #st.write("Prediksi status {}".format(prediction(new_data)))
    # Prediksi model
    raw_prediction = prediction(new_data)  # Output asli dari model (0, 1, atau 2)
    
    # Ubah prediksi menjadi Dropout atau Tidak Dropout
    binary_prediction = map_prediction_to_binary(raw_prediction)
    
    # Tampilkan hasil
    st.write(f"Prediksi status pelajar: **{binary_prediction}**")