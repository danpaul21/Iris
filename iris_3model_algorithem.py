'''
The aim is to classify iris flowers among three species (setosa, versicoloured, or virginica) from measurements of sepals and petals' length and width.

The iris dataset contains 3 classes of 50 instances each, where each class refers to a type of iris plant. The central goal here is to design a model that makes useful classifications for new flowers or, in other words, one which exhibits food generalization. The data source is the file iris_flowers.csv. It contains the data for this example in comma-seperated values (csv) format. The number of columns is 5, and the number of rows is 150.
'''
##IMPORTING LIBRARIES

#IMPORTING ALL OF THE ESSENTIAL LIBRARIES FOR THIS PROJECT

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
#import warnings
#warnings.filterwarnings('ignore')

##LOADING THE DATASET

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
# Load the data
## Explicitly specify the data types for the numerical columns
dtype = {'Sepal length': 'float', 'Sepal width': 'float', 'Petal length': 'float', 'Petal width': 'float', 'Class_labels': 'object'}
## Read the CSV file and skip the first row
df = pd.read_csv('iris_flowers.csv', skiprows= 1, names=columns)
## Display the Data Frame
df.head()

##VISUALISATION OF THE DATASET

df.describe()

# Visualise the whole dataset
sns.pairplot(df, hue='Class_labels')

##SEPERATING INPUT COLUMNS AND THE OUTPUT COLUMN

# Separate features and target
data = df.values
X = data[:,0:4]
Y = data[:,4]
#print(X)
#print(Y)

# Calculate average of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1])
  for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4,3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25

# Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolor')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_ancor=(1.3,1))
plt.show()

##SPLITTING THE DATA INTO TRAINING AND TESTING

# Split the data to train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #, random_state=42)
#print(X_test,Y_test)
#print(X_train, Y_train)

##MODEL1: Support vector machine algorithm

# Support vector machine algorithm
from sklearn.svm import SVC

model_svc = SVC()
model_svc.fit(X_train, Y_train)
#svclassifier = SVC(kernel='linear')
#svclassifier.fit(X_train, y_train)

prediction1 = model_svc.predict(X_test)
# Calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction1)*100)
for i in range(len(prediction1)):
    print(Y_test[i], " ", prediction1[i])
#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, prediction1))
#print(classification_report(y_test, prediction1))

##MODEL2: LOGISTIC REGRESSION

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(X_train, Y_train)

prediction2 = model_LR.predict(X_test)
# Calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction2)*100)
for i in range(len(prediction2)):
    print(Y_test[i], " ", prediction2[i])

##MODEL3: Decision Tree Classifier

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, Y_train)

prediction3 = model_svc.predict(X_test)
# Calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction3)*100)
for i in range(len(prediction3)):
    print(Y_test[i], " ", prediction3[i])

# A detailed classification report
from sklearn.metrics import classification_report #, confusion_matrix
#print(confusion_matrix(y_test, prediction3))
print(classification_report(Y_test, prediction3))

X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
# Prediction of the species from the input vector
prediction = model_svc.predict(X_new)
print("Prediction of Species: {}".format(prediction))
