from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

import seaborn as sns
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#Loading data
data = pd.read_csv(r'train.csv')
testdata = pd.read_csv(r'test.csv')
print(data.describe())
print(data.columns)
num_columns = data.shape[1]
print(f'Number of columns: {num_columns}')
print(data.isnull().sum())

#Replaceing categorical values with numerical values
data['Marital_Status'] = data['Marital_Status'].replace({'Married': 1, 'Single': 0})
data['Radiation_Therapy'] = data['Radiation_Therapy'].replace({'Yes': 1, 'No': 0})
data['Chemotherapy'] = data['Chemotherapy'].replace({'Yes': 1, 'No': 0})
data['Hormone_Therapy'] = data['Hormone_Therapy'].replace({'Yes': 1, 'No': 0})

data['Union_of_therapies'] = ((data['Radiation_Therapy'] == 1) | 
                             (data['Chemotherapy'] == 1) | 
                             (data['Hormone_Therapy'] == 1)).astype(int)
data.insert(9,'Union_of_therapies',data.pop('Union_of_therapies'))

data['TumorandPositiveIntr'] = data['Tumor_Size'] * data['Positive_Axillary_Nodes']
data.insert(10,'TumorandPositiveIntr',data.pop('TumorandPositiveIntr'))

print(data.head())


testdata['Marital_Status'] = testdata['Marital_Status'].replace({'Married': 1, 'Single': 0})
testdata['Radiation_Therapy'] = testdata['Radiation_Therapy'].replace({'Yes': 1, 'No': 0})
testdata['Chemotherapy'] = testdata['Chemotherapy'].replace({'Yes': 1, 'No': 0})
testdata['Hormone_Therapy'] = testdata['Hormone_Therapy'].replace({'Yes': 1, 'No': 0})

testdata['Union_of_therapies'] = ((testdata['Radiation_Therapy'] == 1) | 
                             (testdata['Chemotherapy'] == 1) | 
                             (testdata['Hormone_Therapy'] == 1)).astype(int)
testdata.insert(9,'Union_of_therapies',testdata.pop('Union_of_therapies'))
testdata['TumorandPositiveIntr'] = testdata['Tumor_Size'] * testdata['Positive_Axillary_Nodes']
testdata.insert(10,'TumorandPositiveIntr',testdata.pop('TumorandPositiveIntr'))
print(testdata.head())

#Visualizing relationships between features and the Survival Status
#sns.pairplot(data, hue='Survival_Status',vars=['Age', 'Tumor_Size', 'Positive_Axillary_Nodes',
#'Radiation_Therapy', 'Chemotherapy', 'Hormone_Therapy'])
#plt.title('Pairplot of Features and Survival Status')
#plt.show()


x = data.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values
y = data.iloc[:,-1]

#Spliting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Balancing data
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)


FNNmodel = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)), 
    Dropout(0.5),
    Dense(32, activation='relu'),  
    Dropout(0.5),  
    Dense(1, activation='sigmoid')  
])
FNNmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
FNNmodel.fit(x_train_balanced, y_train_balanced, epochs=50, batch_size=32, validation_split=0.2)

y_pred = FNNmodel.predict(x_test)
y_pred_binary = (y_pred > 0.5).astype(int)
f1 = f1_score(y_test, y_pred_binary)
print(f'F1 Score: {f1:.4f}')
 
# Evaluate the model
#loss, accuracy = FNNmodel.evaluate(x_test, y_test)
#print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
#predictions = FNNmodel.predict(x_test)
#predictions = (predictions > 0.5).astype(int)  

#history = FNNmodel.fit(x_train_balanced, y_train_balanced, epochs=50, batch_size=32, validation_split=0.2)

# Plot training & validation accuracy values
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

# Plot training & validation loss values
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

