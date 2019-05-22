import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('KNN_Project_Data', index_col=0)
print(df.head())

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
df_normalizado = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_normalizado = pd.DataFrame(df_normalizado, columns = df.columns[:-1])
print(df_normalizado)

x_train, x_test, y_train, y_test = train_test_split(df_normalizado, df['TARGET CLASS'], test_size = 0.3)

