# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:58:22 2018

@author: Administrator
"""

import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#读取文件
readFileName="income.xlsx"

#读取excel
data=pd.read_excel(readFileName)
#data=data[['age','workclass','education','sex','hours-per-week','occupation','income']]
data_dummies=pd.get_dummies(data)
print('features after one-hot encoding:\n',list(data_dummies.columns))
features_test=data_dummies.ix[:,"age":'occupation_Transport-moving']
features=data_dummies.ix[:,"age":'native-country_Yugoslavia']
x=features.values
y=data_dummies['income_>50K'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
print("logistic regression:")
print("accuracy on the training subset:{:.3f}".format(logreg.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(logreg.score(x_test,y_test)))
'''
accuracy on the training subset:0.797
accuracy on the test subset:0.797
'''


trees=1000
forest=RandomForestClassifier(n_estimators=trees,random_state=0)
forest.fit(x_train,y_train)

print("random forest with %d trees:"%trees)  
print("accuracy on the training subset:{:.3f}".format(forest.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(forest.score(x_test,y_test)))
print('Feature importances:{}'.format(forest.feature_importances_))
'''
#一百颗数
accuracy on the training subset:1.000
accuracy on the test subset:0.851

#一千颗树
accuracy on the training subset:1.000
accuracy on the test subset:0.853
'''
names=features.columns
importance=forest.feature_importances_
zipped = zip(importance,names)
list1=list(zipped)

list1.sort(reverse=True)
print(list1)

n_features=x.shape[1]
plt.barh(range(n_features),forest.feature_importances_,align='center')
plt.yticks(np.arange(n_features),features.columns)
plt.title("random forest with %d trees:"%trees)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()


