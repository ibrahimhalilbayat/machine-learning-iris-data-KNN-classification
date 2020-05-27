import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import mglearn

print("""
İbrahim Halil Bayat
Department of Electronics and Communication Engineering 
İstanbul Technical University 
İstanbul, Turkey
\n---------- Classifying Iris Species ----------
Species: Setosa, Versicolor, Virginica.
""")

data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=0)
df = pd.DataFrame(x_train, columns=data['feature_names'])

print("\n--------------- 1 Neighbor Classifier ------------------")
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = classifier.predict(x_new)
print("Prediction of {} is {} which means: {}".format(x_new, prediction, data['target_names'][prediction]))

print("\n----------- Prediction in X test -------------")
test_prediction = classifier.predict(x_test)
a = 1
for i in test_prediction:
    print("For sample number {}, class is {} which means {}. ".format(a, i, data['target_names'][i]))
    a += 1


the_predictions = pd.DataFrame()
the_predictions['Samples'] = y_test
the_predictions['Predicted'] = test_prediction
the_predictions['Predicted-Values'] = [data['target_names'][i] for i in test_prediction]
the_predictions['Actual-Values'] = [data['target_names'][i] for i in y_test]


print(the_predictions)
the_predictions.to_csv("predictions.csv", index=False)

print("\n--------- Evaluating the Model ---------------")
print("Accuracy: ", np.mean(y_test == test_prediction))

print("\n---------------------- The Score ---------------------")
print("The Score: ", classifier.score(x_test, y_test))