# -*- coding: utf-8 -*-
"""
SCIKIT－LEARN 的例子。“统一模式”
"""
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split  
from sklearn.neighbors import KNeighborsClassifier

#1.------------准备数据------------------------
iris = datasets.load_iris()   #使用sk的数据库。 SK本身就有很多数据库
iris_X = iris.data            #X 是输入
iris_y = iris.target          #y 是结果
#print(iris_X[:5,:])
#print(iris_y)

#2.------------数据分组------------------------
X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3) #随机分数据为训练集和数据集

#3.------------训练算法------------------------
knn=KNeighborsClassifier()  #初始化分类器（分类器）
knn.fit(X_train,y_train)    #训练分类器

#4.------------表示结果------------------------
predictValue =knn.predict(X_test) #预测值
actuleValue = y_test              #实际值
print(predictValue==actuleValue)
print('预测:',predictValue)    #显示预测值
print('实际:',actuleValue)     #显示实际值
