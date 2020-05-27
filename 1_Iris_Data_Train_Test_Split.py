import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("""
İbrahim Halil Bayat
Department of Electronics and Communication Engineering 
İstanbul Technical University 
İstanbul, Turkey
\n---------- Classifying Iris Species ----------
Species: Setosa, Versicolor, Virginica.
""")

print("\n------------------ Uploading the Data Set from 'sklearn.datasets' ----------")
from sklearn.datasets import load_iris

data = load_iris()
print("What the data has: ", data.keys())
print("DESCR of the data: ", data['DESCR'][: 193])
print("\nTarget names: ", data['target_names'])
print("\nFeature Names: ", data['feature_names'])

print("\n------------------ Type of the Data ------------------")
print("Type of the data: ", type(data['data']))

print("\n--------------- Shape of the Data --------------------")
print("Shape of the Data: ", data['data'].shape)

print("\n--------- First Five Columns of the Data -------------")
print(data['data'][:5])

print("\n---------------- Train Test Split --------------------")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=0)
print("Shape of x train: ", x_train.shape)
print("Shape of x test: ", x_test.shape)
print("Shape of y train: ", y_train.shape)
print("Shape of y test: ", y_test.shape)

