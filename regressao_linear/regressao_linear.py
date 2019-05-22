import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('USA_Housing.csv')

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(x_train, y_train)
predict = lm.predict(x_test)

print(lm.score(x_test, y_test))

