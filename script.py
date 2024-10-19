from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#Loading data
data = pd.read_csv(r'train.csv')
testdata = pd.read_csv(r'test.csv')

print(data.columns)
num_columns = data.shape[1]
print(f'Number of columns: {num_columns}')
print(data.isnull().sum())

data['Marital_Status'] = data['Marital_Status'].replace({'Married': 1, 'Single': 0})
data['Radiation_Therapy'] = data['Radiation_Therapy'].replace({'Yes': 1, 'No': 0})
data['Chemotherapy'] = data['Chemotherapy'].replace({'Yes': 1, 'No': 0})
data['Hormone_Therapy'] = data['Hormone_Therapy'].replace({'Yes': 1, 'No': 0})

data['TumorandPositiveIntr'] = data['Tumor_Size'] * data['Positive_Axillary_Nodes']
data.insert(9,'TumorandPositiveIntr',data.pop('TumorandPositiveIntr'))
data['TumorandAgeintr'] = data['Tumor_Size'] * data['Age']
data.insert(10,'TumorandAgeintr',data.pop('TumorandAgeintr'))



print(data.head())
print(data.columns)

testdata['Marital_Status'] = testdata['Marital_Status'].replace({'Married': 1, 'Single': 0})
testdata['Radiation_Therapy'] = testdata['Radiation_Therapy'].replace({'Yes': 1, 'No': 0})
testdata['Chemotherapy'] = testdata['Chemotherapy'].replace({'Yes': 1, 'No': 0})
testdata['Hormone_Therapy'] = testdata['Hormone_Therapy'].replace({'Yes': 1, 'No': 0})

testdata['TumorandPositiveIntr'] = testdata['Tumor_Size'] * testdata['Positive_Axillary_Nodes']
testdata.insert(9,'TumorandPositiveIntr',testdata.pop('TumorandPositiveIntr'))
testdata['TumorandAgeintr'] =testdata['Tumor_Size'] * testdata['Age']
testdata.insert(10,'TumorandAgeintr',testdata.pop('TumorandAgeintr'))


print(testdata.columns)
print(testdata.head())



x = data.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values
y= data.iloc[:,-1]
x_test = testdata.iloc[:, [1,2,3,4,5,6,7,8,9,10]].values




print(x)
print(y)

print(data)
print(data.dtypes)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

#Feature scaling

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.transform(x_val)
x_test = sc.transform(x_test)

#Balancing data
print('before',pd.Series(y_train).value_counts())
smote = SMOTE(random_state=0)
x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
print('after', pd.Series(y_resampled).value_counts()) 

#Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, scoring='f1', cv=5)
grid_search.fit(x_resampled, y_resampled)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best F1 Score: {best_score}")
print(f"Best Parameters: {best_params}")

#Training the model

classifier = RandomForestClassifier(**best_params)
classifier.fit(x_resampled, y_resampled)
importances = classifier.feature_importances_
feature_names = ['Age', 'Marital_Status', 'Year of Operation', 
                 'Positive_Axillary_Nodes', 'Tumor_Size', 
                 'Radiation_Therapy', 'Chemotherapy', 
                 'Hormone_Therapy', 'TumorandPositiveIntr', 
                 'TumorandAgeintr']

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance

plt.figure(figsize=(15, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest Model')
plt.show()

score = classifier.score(x_resampled, y_resampled)
print (score)
y_test_pred = classifier.predict(x_test)
y_val_pred = classifier.predict(x_val)
f1 = f1_score(y_val, y_val_pred)
print(f'F1 Score on validation set: {f1}')

#Submission

submission_df = pd.DataFrame({
    'Patient_ID': testdata['Patient_ID'],
    'Predicted_Survival_Status': y_test_pred
})


submission_df.to_csv('submission.csv', index=False)






