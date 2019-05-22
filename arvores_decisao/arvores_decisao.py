import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('kyphosis.csv')
print(df.info())
print(df.head())

x_train, x_test, y_train, y_test = train_test_split(df.drop('Kyphosis', axis=1), df['Kyphosis'], test_size = 0.3)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
predict = decision_tree.predict(x_test)

