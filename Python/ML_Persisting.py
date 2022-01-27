import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# from sklearn.externals import joblib
import joblib

# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns=['genre'])
# y = music_data['genre']

# model = DecisionTreeClassifier()
# model.fit(X, y)

# joblib.dump(model, 'music_recommender.joblib')
#---------------------

model = joblib.load('music_recommender.joblib')
predictions = model.predict([ [21,1] ])

print(predictions)