import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('loan_data.csv')
print(df.info())
print(df.head())

cat_feats = ['purpose']
final_df = pd.get_dummies(df, columns=cat_feats, drop_first=True)
print(final_df.info())


x_train, x_test, y_train, y_test = train_test_split(final_df.drop('not.fully.paid', axis=1), final_df['not.fully.paid'], test_size=0.3, random_state=101)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
predictions = decision_tree.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
