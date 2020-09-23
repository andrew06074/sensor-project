import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Import data from csv files
def load_metadata(nrows):
    metadata = pd.read_csv('metadata.csv', nrows=nrows)
    metadata.set_index("Unnamed: 0",inplace=True)
    return metadata

def load_data(nrows):
    df = pd.read_csv('sensor-sl.csv', nrows=nrows)
    df.set_index("Unnamed: 0",inplace=True)
    return df

# Display data loading text
df = load_data(200000)
metadata = load_metadata(200000)


# Title of app
st.title('Please select an item from the sidebar')

# Create selectionbox for events
ids = [] 
for item in metadata['Id']:
    x = item
    ids.append(x)

# Create list of data attributes (Column names)
att_list = list(df.columns)

#SIDEBAR OPTIONS
# Selected item is stored as event_name
event_name = st.sidebar.selectbox("Select item :", ids)

# Create selection box for classifiers
clf_name = st.sidebar.selectbox("Select classifier", ("KKN","Random Forest"))

def get_event(event_name,df):
    event = df.loc[df['Id']== event_name]
    return event

event = get_event(event_name,df)

# Checkbox for selected id
event_name_str = str(event_name)
if st.checkbox('Show data for ID:'+ event_name_str):
    event.set_index("Unnamed: 0",inplace=True)
    st.write(event)

# Plot selected item
fig, axs = plt.subplots(5,2, figsize=(7,7))
# X for graph
X_data = event['Time']
# Y for graph
sens1_data = event['Sens-1']
sens2_data = event['Sens-2']
sens3_data = event['Sens-3']
sens4_data = event['Sens-4']
sens5_data = event['Sens-5']
sens6_data = event['Sens-6']
sens7_data = event['Sens-7']
sens8_data = event['Sens-8']
temp_data = event['Temp']
Co2_data = event['Co2']

# Plot data
axs[0,0].plot(X_data,sens1_data,'tab:orange')
axs[0,0].set_title('Sensor 1')
axs[0,1].plot(X_data,sens2_data,'tab:orange')
axs[0,1].set_title('Sensor 2')
axs[1,0].plot(X_data,sens3_data,'tab:orange')
axs[1,0].set_title('Sensor 3')
axs[1,1].plot(X_data,sens4_data,'tab:orange')
axs[1,1].set_title('Sensor 4')
axs[2,0].plot(X_data,sens5_data,'tab:orange')
axs[2,0].set_title('Sensor 5')
axs[2,1].plot(X_data,sens6_data,'tab:orange')
axs[2,1].set_title('Sensor 6')
axs[3,0].plot(X_data,sens7_data,'tab:orange')
axs[3,0].set_title('Sensor 7')
axs[3,1].plot(X_data,sens8_data,'tab:orange')
axs[3,1].set_title('Sensor 8')
axs[4,0].plot(X_data,temp_data,'tab:red')
axs[4,0].set_title('Temp ( C*)')
axs[4,1].plot(X_data,Co2_data)
axs[4,1].set_title('Co2')

fig.tight_layout()
st.pyplot(fig)

# Display sensor data, hidden by default
st.title('Select a model and parameters to test')
if st.checkbox('Show raw sensor data'):
    st.dataframe(df,750,300)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KKN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    else:
        max_depth = st.sidebar.slider("max_depth", 2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(clf_name)

def get_classifier(clf_name,params):
    if clf_name == "KKN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    else:
      clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1530)
    return clf

clf = get_classifier(clf_name,params)



# Classification
X = df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11]].values
y = df.iloc[:, 12].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1530)

# Scalse model
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Reports
acc = accuracy_score(y_test, y_pred) * 100
clf_report = classification_report(y_test,y_pred)

# Output information
st.subheader("Model information")
st.write(f"Selected classifier = {clf_name}")
st.write(f"Model accuracy = {acc}")
st.subheader("Confusion matrix")
st.write(clf_report)

# Plot title
st.subheader("Displaying prediction model")
st.write("Changing the components on the sidebar will change values of the" + '\n' +
"plot however in this visulization we are only looking at the predicted Temp and Co2.")
st.write("Purple --- Predicted event")
st.write("Red --- Predicted Non-event")
pca = PCA(10)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 8]
x2 = X_projected[:, 9]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.7, cmap="Spectral")
plt.xlabel('Scaled Component (Temp(C*))')
plt.ylabel('Scaled Component (Co2)')
plt.colorbar()

st.pyplot(fig)

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
