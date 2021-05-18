import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Hospital Discharge Time Prediction App
This app predicts the hospital stay time!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:

    def user_input_features():
        Hospital_type_code = st.sidebar.selectbox('Hospital_type_code', ('a', 'b', 'c', 'd', 'e', 'f', 'g'))
        City_Code_Hospital = st.sidebar.slider('City_Code_Hospital', 1, 13, 5, 1)
        Hospital_region_code = st.sidebar.selectbox('Hospital_region_code', ('X', 'Y', 'Z'))
        Available_Extra_Rooms_in_Hospital = st.sidebar.slider('Available.Extra.Rooms.in.Hospital', 0, 24, 3, 1)
        Department = st.sidebar.selectbox('Department',('anesthesia', 'gynecology', 'radiotherapy', 'surgery', 'TB & Chest disease'))
        Ward_Type = st.sidebar.selectbox('Ward_Type', ('P', 'Q', 'R', 'S', 'T', 'U'))
        Ward_Facility_Code = st.sidebar.selectbox('Ward_Facility_Code', ('A', 'B', 'C', 'D', 'E', 'F'))
        Bed_Grade = st.sidebar.slider('Bed.Grade', 1, 4, 3, 1)
        Type_of_Admission = st.sidebar.selectbox('Type.of.Admission', ('Emergency', 'Trauma', 'Urgent'))
        Severity_of_Illness = st.sidebar.selectbox('Severity.of.Illness', ('Extreme', 'Minor', 'Moderate'))
        Visitors_with_Patient = st.sidebar.slider('Visitors.with.Patient', 0, 32, 3, 1)
        Age = st.sidebar.selectbox('Age', ('0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'))
        Admission_Deposit = st.sidebar.slider('Admission_Deposit', 1800, 11008, 4741, 1)

        data = {'Hospital_type_code': Hospital_type_code,
            'City_Code_Hospital': City_Code_Hospital,
            'Hospital_region_code': Hospital_region_code,
            'Available.Extra.Rooms.in.Hospital': Available_Extra_Rooms_in_Hospital,
            'Department': Department,
            'Ward_Type': Ward_Type,
            'Ward_Facility_Code': Ward_Facility_Code,
            'Bed.Grade': Bed_Grade,
            'Type.of.Admission': Type_of_Admission,
            'Severity.of.Illness': Severity_of_Illness,
            'Visitors.with.Patient': Visitors_with_Patient,
            'Age': Age,
           'Admission_Deposit': Admission_Deposit}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
hospital_raw = pd.read_csv('hospital_discharge_data.csv')
hospital = hospital_raw.drop(columns=['Stay'])
df = pd.concat([input_df, hospital],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Hospital_type_code','Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type.of.Admission', 'Severity.of.Illness', 'Age']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('Discharge_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
hospital_Stay = np.array(['0-10','11-20','21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', 'More than 100 Days'])
st.write(hospital_Stay[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)