import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
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

print("\n--------- Converting NumPy into Pandas ---------------")
df = pd.DataFrame(x_train, columns=data['feature_names'])
grr = pd.plotting.scatter_matrix(df, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8,
                        cmap=mglearn.cm3)
plt.show()