from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Loading data
data = pd.read_csv(r'train.csv')
testdata = pd.read_csv(r'test.csv')

data['Marital_Status'] = data['Marital_Status'].replace({'Married': 1, 'Single': 0})
data['Radiation_Therapy'] = data['Radiation_Therapy'].replace({'Yes': 1, 'No': 0})
data['Chemotherapy'] = data['Chemotherapy'].replace({'Yes': 1, 'No': 0})
data['Hormone_Therapy'] = data['Hormone_Therapy'].replace({'Yes': 1, 'No': 0})

data['TumorandPositiveIntr'] = data['Tumor_Size'] * data['Positive_Axillary_Nodes']
data.insert(9, 'TumorandPositiveIntr', data.pop('TumorandPositiveIntr'))
data['TumorandAgeintr'] = data['Tumor_Size'] * data['Age']
data.insert(10, 'TumorandAgeintr', data.pop('TumorandAgeintr'))

print(data.head())
print(data.columns)

testdata['Marital_Status'] = testdata['Marital_Status'].replace({'Married': 1, 'Single': 0})
testdata['Radiation_Therapy'] = testdata['Radiation_Therapy'].replace({'Yes': 1, 'No': 0})
testdata['Chemotherapy'] = testdata['Chemotherapy'].replace({'Yes': 1, 'No': 0})
testdata['Hormone_Therapy'] = testdata['Hormone_Therapy'].replace({'Yes': 1, 'No': 0})

testdata['TumorandPositiveIntr'] = testdata['Tumor_Size'] * testdata['Positive_Axillary_Nodes']
testdata.insert(9, 'TumorandPositiveIntr', testdata.pop('TumorandPositiveIntr'))
testdata['TumorandAgeintr'] = testdata['Tumor_Size'] * testdata['Age']
testdata.insert(10, 'TumorandAgeintr', testdata.pop('TumorandAgeintr'))

print(testdata.columns)
print(testdata.head())

# Splitting data into features and target
x = data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
y = data.iloc[:, -1]
x_test = testdata.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values

print(x)
print(y)

# Splitting data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.transform(x_val)
x_test = sc.transform(x_test)

# SMOTE for balancing the dataset
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

# Building the Neural Network model
FNNmodel = Sequential([
    Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu',),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compiling the model
FNNmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model and storing history for plotting
history = FNNmodel.fit(x_train_balanced, y_train_balanced, epochs=50, batch_size=32, validation_split=0.2)

# Predict on validation set
y_pred = FNNmodel.predict(x_val)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate F1 score using the validation labels
f1 = f1_score(y_val, y_pred_binary)
print(f'F1 Score: {f1}')

# Accuracy on validation set
loss, accuracy = FNNmodel.evaluate(x_val, y_val)
print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predict on the test set
y_pred_test = FNNmodel.predict(x_test)
y_pred_binary_test = (y_pred_test > 0.5).astype(int)

# Prepare the submission file
y_pred_binary_test = y_pred_binary_test.ravel()
submission_df = pd.DataFrame({
    'Patient_ID': testdata['Patient_ID'],
    'Predicted_Survival_Status': y_pred_binary_test
})

submission_df.to_csv('submissionfnn.csv', index=False)