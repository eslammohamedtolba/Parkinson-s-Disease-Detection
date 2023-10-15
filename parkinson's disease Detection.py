# Import required modules
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


# Loading the dataset
parkinson_disease_dataset = pd.read_csv("parkinsons.csv")
# How the first five samples dataset
parkinson_disease_dataset.head()
# Show the last five samples dataset
parkinson_disease_dataset.tail()
# Show the dataset shape
parkinson_disease_dataset.shape
# Show some statistical info about the dataset
parkinson_disease_dataset.describe()
# Find the mean between the features and the output
parkinson_disease_dataset.groupby('status').mean()



# Check about any missing(none) values in the dataset if will make a data cleaning or not
parkinson_disease_dataset.isnull().sum()

# Count the values in the status column and its repetition
parkinson_disease_dataset['status'].value_counts()
# Catplot the status column values and its repetition
plt.figure(figsize=(5,5))
sns.catplot(x = 'status',data=parkinson_disease_dataset,kind='count')



# Split data into input and label data
X = parkinson_disease_dataset.drop(columns=['status','name'],axis=1)
Y = parkinson_disease_dataset['status']
# Show the serapate data
print(X)
print(Y)
# Make a standard scalling for input data to have a common range
sc = StandardScaler()
X = sc.fit_transform(X)
# Show the input data after scalling process
print(X)
# Split the data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)


# Create and train the model
svcModel = svm.SVC()
svcModel.fit(x_train,y_train)
# Make the model predict the train and test data
predicted_train_data = svcModel.predict(x_train)
predicted_test_data = svcModel.predict(x_test)
# Avaluate the model
accuracy_train = accuracy_score(predicted_train_data,y_train)
accuracy_test = accuracy_score(predicted_test_data,y_test)
print(accuracy_train,accuracy_test)


# Makeing a predictive system
input_data=(120.08000,139.71000,111.20800,0.00405,0.00003,0.00180,0.00220,0.00540,0.01706,0.15200,0.00974,0.00925,0.01345,0.02921,0.00442,25.74200,0.495954,0.762959,-5.791820,0.329066,2.205024,0.188180)
# Convert input data into 1D numpy array
input_array = np.array(input_data)
# Convert 1D input array into 2D
input_2D_array = input_array.reshape(1,-1)
input_2D_array = sc.fit_transform(input_2D_array)
# Predict the input
if svcModel.predict(input_2D_array)[0]==1:
    print("this person has parkinson disease")
else:
    print("this person doesn't have parkinson disease")


