import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

pd.set_option('display.max_columns', None)

df = pd.read_csv('ecommerce_customers.csv')
print(df.head())
print(df.info())
print(df.describe())

x = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = df['Yearly Amount Spent']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)

lm = LinearRegression()
lm.fit(x_train,y_train)
print(lm.coef_)
print(lm.intercept_)
print(x_test)
predict = lm.predict(x_test)

print("MAE: ", metrics.mean_absolute_error(y_test, predict))
print("MSE: ", metrics.mean_squared_error(y_test, predict))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predict)))

sns.distplot(y_test-predict)
plt.show()
