# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split

from sklearn import metrics
data = pd.read_csv('C:/Users/Sudhish Subramaniam/Desktop/test1.csv') 
Ans=[]
a=int(input())
for i in range(1,10):
    
    b=int(input())
    s1=pd.concat([data['date'],data['f'+str(i)],data['f'+str(i)+'b']],axis=1, keys=['date', 'f'+str(i),'f'+str(i)+'b'])
    X = s1.iloc[:, 0:2].values
    y = s1.iloc[:, 2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=0)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    Y=model.predict([[a,b]])
    print(Y)
    Ans.append(Y)
