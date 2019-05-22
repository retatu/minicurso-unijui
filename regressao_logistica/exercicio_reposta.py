import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('advertising.csv')
x_train, x_test, y_train, y_test = train_test_split(df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']], df['Clicked on Ad'], test_size=0.3)

lg = LogisticRegression()
lg.fit(x_train, y_train)
predictions = lg.predict(x_test)

print(lg.score(x_test, y_test))
