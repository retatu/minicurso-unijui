import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

train = pd.read_csv('titanic_train.csv')
print(train.head())
print(train.info())

print(train.isnull().sum())

def set_idade(cols):
    idade = cols[0]
    pclass = cols[1]
    if pd.isnull(idade):
        if pclass == 1:
            return 37
        if pclass == 2:
            return 29
        else:
            return 24
    else:
        return idade
train['Age'] = train[['Age', 'Pclass']].apply(set_idade, axis=1)
del train['Cabin']
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'], drop_first=True)
embarked= pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex','PassengerId','Name','Ticket', 'Embarked'], axis=1, inplace=True)
train = pd.concat([train, sex, embarked], axis=1)


x_train, x_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1), train['Survived'], test_size=0.3)
lg = LogisticRegression()
lg.fit(x_train, y_train)
predictions = lg.predict(x_test)



