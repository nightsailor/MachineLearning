## 1 import the data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
music_data

## 2 Clean the data
## 3 Split the data into Training/Test Sets
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## 4 Create the model
# Algorithm - decision tree
modelA DecisionTreeClassifier()
model = DecisionTreeClassifier()

## 5 Train the model
modelA.fit(X, y)
model.fit(X_train, y_train)
#no value for 22 F

## 6 Make Predictions
predictionsA = model.predict([ [21,1], [22,0] ])
predictions = model.predict(X_test)

## 7 Evaluate and improve
# Calculating the Accuracy
score = accuracy_score(y_test, predictions)

print(score)
