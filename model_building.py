import pandas as pd
discharge = pd.read_csv('hospital_discharge_data.csv')

# Ordinal feature encoding

df = discharge.copy()
target = 'Stay'
encode = ['Hospital_type_code','Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type.of.Admission', 'Severity.of.Illness', 'Age']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'0-10':0, '11-20':1, '21-30':2, '31-40':3, '41-50':4, '51-60':5, '61-70':6, '71-80':7, '81-90':8, '91-100':9, 'More than 100 Days':10}
def target_encode(val):
    return target_mapper[val]

df['Stay'] = df['Stay'].apply(target_encode)

# Separating X and y
X = df.drop('Stay', axis=1)
Y = df['Stay']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('Discharge_clf.pkl', 'wb'))
