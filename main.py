import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,f1_score
import joblib
import streamlit as st


# Attempt to load the data again to ensure the DataFrame is not empty
try:
    data = pd.read_csv('/Users/nayanpaliwal/Desktop/The Customer Churn Prediction/dataset/telco.csv')
    print("Data loaded successfully. First 5 rows:")
    print(data.head())
    print("\nShape of the loaded data:", data.shape)
except FileNotFoundError:
    print("Error: 'telco.csv' not found. Please ensure the file is in the correct location.")
    data = pd.DataFrame() # Assign an empty DataFrame if file not found

import numpy as np
missing_mask = data.isnull()

missing_per_column = missing_mask.sum(axis=0)
print("missing values per column",missing_per_column)
total_missing = missing_mask.sum().sum()
print("total missing values",total_missing)

# Dealing with Null values 

offer_mode= data['Offer'].mode()[0]


internet_type_mode=data['Internet Type'].mode()[0]
data['Internet Type']=data['Internet Type'].fillna(internet_type_mode)

churn_category_mode=data['Churn Category'].mode()[0]
data['Churn Category']=data['Churn Category'].fillna(churn_category_mode)

churn_reason_mode=data['Churn Reason'].mode()[0]
data['Churn Reason']=data['Churn Reason'].fillna(churn_reason_mode)

#Time to plot

data['Age'].plot(kind='hist', bins=20, title='Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

data['Senior Citizen'].value_counts().plot(kind='pie')

sns.barplot(data,x=data['Monthly Charge'],y=data['Churn Category'])

data['Internet Type'].value_counts().plot(kind='pie')

sns.barplot(data,x=data['Customer Status'],y=data['Monthly Charge'])

sns.barplot(data, x=data['Churn Category'], y=data['Number of Dependents'])

fig, axes = plt.subplots(1, 1, figsize=(10, 6))
sns.histplot(data=data, x="Monthly Charge", hue="Churn Label", kde=True, ax=axes, bins=20)
axes.set_title("Monthly Charge Distribution by Churn")
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(10, 6))
sns.histplot(data=data, x="Tenure in Months", hue="Churn Label", kde=True, ax=axes, bins=20)
axes.set_title("Monthly Charge Distribution by Churn")
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(10, 6))
sns.histplot(data=data, x="Satisfaction Score", hue="Churn Label", kde=True, ax=axes, bins=20)
axes.set_title("Monthly Charge Distribution by Churn")
plt.show()

cat_columns = ["Contract", "Payment Method", "Offer"]

# Set up subplots
fig, axes = plt.subplots(1, len(cat_columns), figsize=(18, 5))

for i, col in enumerate(cat_columns):
    sns.countplot(data=data, x=col, hue="Churn Label", ax=axes[i])
    axes[i].set_title(f"{col} Count by Churn")
    axes[i].tick_params(axis='x', rotation=45)

#Preprocessing for few columns

data['Customer Status']= data['Customer Status'].map({"Churned":0,"Stayed":1,"Joined":2})
data['Gender']=data['Gender'].map({'Male':1,'Female':0})

# Map 'Churn Label' to numerical values before splitting the data
data["Churn Label"] = data["Churn Label"].map({"Yes":1,"No":0})

# Drop rows where 'Churn Label' is NaN
data.dropna(subset=["Churn Label"], inplace=True)

X = data.drop(columns=['Customer ID', 'Offer', 'Churn Category', 'Churn Reason', 'Churn Score', 'Churn Label','Population', 'State', 'CLTV', 'Online Security', 'Streaming TV',
                       'Married', 'Unlimited Data', 'Under 30', 'Total Refunds', 'Quarter', 'Country', 'Referred a Friend', 'Streaming Movies', 'Latitude', 'Avg Monthly GB Download', 'City', 'Multiple Lines', 'Total Long Distance Charges', 'Avg Monthly Long Distance Charges', 
                       'Online Backup', 'Internet Service', 'Dependents', 'Premium Tech Support','Total Extra Data Charges', 'Internet Type', 'Senior Citizen', 'Customer Status', 'Longitude', 'Zip Code', 'Phone Service', 'Streaming Music', 'Total Revenue', 'Satisfaction Score', 'Device Protection Plan', 'Gender', 'Paperless Billing'])
y = data["Churn Label"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include='object').columns
numerical_cols = X_train.select_dtypes(include=np.number).columns

# Create transformers for one-hot encoding and scaling
preprocessor = ColumnTransformer(
    transformers=[
        # Add SimpleImputer for numerical columns before scaling
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# Create a pipeline that first preprocesses and then scales
# Note: StandardScaler is applied to numerical columns within the preprocessor
# So we don't need a separate scaling step here.
# We can directly transform the data using the preprocessor.

x_trained_scaled = preprocessor.fit_transform(X_train)
x_test_scaled = preprocessor.transform(X_test)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
#print(classification_report(y_test,y_pred))

#   precision    recall  f1-score   support

#           0       1.00      1.00      1.00      1009
#          1       1.00      1.00      1.00       400

#   accuracy                           1.00      1409
#  macro avg       1.00      1.00      1.00      1409
#weighted avg       1.00      1.00      1.00      1409

model2 = RandomForestClassifier()
model2.fit(x_trained_scaled,y_train)
y_pred = model2.predict(x_test_scaled)
#print(classification_report(y_test,y_pred))

#    precision    recall  f1-score   support

#           0       0.98      1.00      0.99      1009
#          1       1.00      0.95      0.98       400

#   accuracy                           0.99      1409
#  macro avg       0.99      0.98      0.98      1409
#weighted avg       0.99      0.99      0.99      1409

#As we can see the model1 has better results, so we will be moving ahead with Logistic Regression model 

joblib.dump(pipeline,'model.joblib')
load_model=joblib.load('model.joblib')
import os
print(os.getcwd())


