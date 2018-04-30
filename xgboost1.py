# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 22:42:47 2018

@author: Administrator
出现module 'xgboost' has no attribute 'DMatrix'的临时解决方法
初学者或者说不太了解Python才会犯这种错误，其实只需要注意一点！不要使用任何模块名作为文件名，任何类型的文件都不可以！我的错误根源是在文件夹中使用xgboost.*的文件名，当import xgboost时会首先在当前文件中查找，才会出现这样的问题。
        所以，再次强调：不要用任何的模块名作为文件名！
"""
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import pandas as pd
import matplotlib.pylab as plt

#读取文件
readFileName="income.xlsx"

#读取excel
data=pd.read_excel(readFileName)
#data=data[['age','workclass','education','sex','hours-per-week','occupation','income']]
data_dummies=pd.get_dummies(data)
print('features after one-hot encoding:\n',list(data_dummies.columns))
features=data_dummies.ix[:,"age":'native-country_Yugoslavia']
x=features.values
y=data_dummies['income_>50K'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
names=features.columns


dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)

params={'booster':'gbtree',
    #'objective': 'reg:linear',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

ypred=bst.predict(dtest)

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1

#模型校验
from sklearn import metrics
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)
'''
AUC: 0.9107
ACC: 0.8547
Recall: 0.5439
F1-score: 0.6457
Precesion: 0.7944
Out[28]: 
array([[5880,  279],
       [ 904, 1078]], dtype=int64)
'''


print("xgboost:")  
print('Feature importances:{}'.format(bst.get_fscore()))


'''
Feature importances:{'f33': 76, 'f3': 273, 'f4': 157, 'f25': 11, 'f0': 167,
 'f42': 34, 'f2': 193, 'f5': 132, 'f56': 1, 'f64': 14, 'f24': 11, 'f53': 15,
 'f58': 24, 'f39': 2, 'f1': 20, 'f29': 3, 'f35': 9, 'f48': 20, 'f12': 11, 
 'f65': 3, 'f27': 3, 'f50': 3, 'f26': 7, 'f60': 2, 'f43': 8, 'f85': 1,
 'f10': 1, 'f46': 5, 'f11': 1, 'f49': 1, 'f7': 1, 'f52': 3, 'f66': 1, 
 'f54': 1, 'f23': 1}
'''
