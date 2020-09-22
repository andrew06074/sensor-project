import streamlit as st


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Import data from csv files
def load_metadata(nrows):
    metadata = pd.read_csv('metadata.csv', nrows=nrows)
    return metadata

def load_data(nrows):
    df = pd.read_csv('sensor-sl.csv', nrows=nrows)
    return df

# Display data loading text
data_load_state = st.text('Data loading, please wait')
df = load_data(10000)
metadata = load_metadata(10000)
data_load_state.text("Data loading: Done!")

# Title of app
st.title('Sensor data applied to Otis')

# Display metadata, hidden by default
st.subheader('Sensor Metadata')
if st.checkbox('Show raw metadata'):
    metadata.set_index("Unnamed: 0",inplace=True)
    st.dataframe(metadata,750,300)
st.text('Here is the breakdown of events by type')
st.text('Event --- Event occurs, this is the event we are looking to classify')
st.text('Non_Event --- False flag, not the same type of event')
st.text('Null_Event --- Iteration of non induction, nothing was added to the test enviorment')

# Create count chart of items
counts = metadata['Description'].value_counts()
st.bar_chart(counts)
st.write(counts)

# Display sensor data, hidden by default
st.subheader('Sensor Dataset')
if st.checkbox('Show raw sensor data'):
    st.dataframe(df,750,300)

# Create selectionbox for events
ids = [] 
for item in metadata['Id']:
    x = item
    ids.append(x)

# Selected item is stored as event_name
event_name = st.selectbox("Select event :", ids)

def get_event(event_name,df):
    event = df.loc[df['Id']== event_name]
    return event

event = get_event(event_name,df)
event.set_index("Unnamed: 0",inplace=True)
st.write(event)

# Create selection box for classifiers
clf_name = st.sidebar.selectbox("Select Classifier", ("KKN","SVM","Random Forest"))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KKN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01,10)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2,15)
        n_estimators = st.sidebar.slider("n_estimators")
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


params = add_parameter_ui(clf_name)

def get_classifier(clf_name,params):
    if clf_name == "KKN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
      clf = RandomForestClassifer(n_neighbors=params["n_estimators"], max_depth=params["max_depth"], random_state=1530)
    return clf

clf = get_classifier(clf_name,params)


# Classification
X = df.data
y= data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1530)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier = {clf_name}")
st.write(f"accuracy = {acc}")
