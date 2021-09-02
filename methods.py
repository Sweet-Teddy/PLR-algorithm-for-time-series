# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 08:10:18 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt  ##绘图库
from scipy.optimize import leastsq  ##引入最小二乘法算法
#-----------------------最小二乘函数---------------------
def min_error_func(Xi,Yi):
    #先将传进来的数据进行标准化，去除量纲的影响
    
    ##需要拟合的函数func :指定函数的形状
    def error(p,x,y):
        return func(p,x)-y
     
    ##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
    def func(p,x):
        k,b=p
        return k*x+b  
    '''
    主要部分：附带部分说明
    1.leastsq函数的返回值tuple，第一个元素是求解结果，第二个是求解的代价值(个人理解)
    2.官网的原话（第二个值）：Value of the cost function at the solution
    3.实例：Para=>(array([ 0.61349535,  1.79409255]), 3)
    4.返回值元组中第一个值的数量跟需要求解的参数的数量一致
'''
    #k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
    p0=[1,20]   
    #把error函数中除了p0以外的参数打包到args中(使用要求)
    Para=leastsq(error,p0,args=(Xi,Yi))

    #读取结果
    k,b=Para[0]
# =============================================================================
#     print("k=",k,"b=",b)
#     print("cost："+str(Para[1]))
#     print("求解的拟合直线为:")
#     print("y="+str(round(k,2))+"x+"+str(round(b,2)))
# =============================================================================

    '''
   绘图，看拟合效果.
   matplotlib默认不支持中文，label设置中文的话需要另行设置
   如果报错，改成英文就可以
   '''

    #画样本点
   # plt.figure(figsize=(8,6)) ##指定图像比例： 8：6
# =============================================================================
#     plt.figure(figsize=(8,6)) ##指定图像比例： 8：6
#     plt.scatter(Xi,Yi,color="green",label="demo_data",linewidth=2) 
# 
#     #画拟合直线
#     x=np.linspace(-2,2,100) ##在0-15直接画100个连续点
#     y=k*x+b ##函数式
# =============================================================================
# =============================================================================
#     plt.plot(x,y,color="red",label="approximate_line",linewidth=2) 
#     plt.legend(loc='lower right') #绘制图例
#     plt.show()
# =============================================================================
    return k,b#返回直线的系数k和b

#-----------------------日期戳转化为时间戳数组函数------------------------
def time_translate(Xi):
    #a1 = "2019-5-10 23:40:00"
    # 先转换为时间数组
    #timeArray = time.strptime(Xi, "%Y-%m-%d %H:%M:%S")
    timeArray = []  
    timeStamp = []
    for i in range(len(Xi)):
        timeArray.append(time.strptime(Xi[i], "%Y/%m/%d"))
        # 转换为时间戳
        timeStamp.append(int(time.mktime(timeArray[i])))
    #print(timeStamp)
    return np.array(timeStamp)

#-----------------------加载数据函数--------------------
def loadDataSet(fileName,encoding):
    dataSetList = []
    df = pd.read_csv(fileName,encoding='utf-8')
    #dataSetList = df.drop(['Date'], axis = 1)
    dataSetList = df
    return dataSetList


#----------------------计算拟合误差---------------------
#用插值直线与原数据的均方差来表示error
#参数描述： x1：分段点的时间戳  y1: 分段点对应的y值  ts:原始时间序列的y值  X:原始时间序列的时间戳
def calculate_fitting_error(x1,y1,ts,X):
    sum_error = 0
    for i in range(len(y1)-1):
        k = (y1[i+1] - y1[i])/(x1[i+1] - x1[i])
        b = y1[i] - k*x1[i]
        #得到插值直线的斜率和截距 即：y = kx + b
        count = 0
        error1= 0
        j = ts.tolist().index(y1[i])
        m = ts.tolist().index(y1[i+1])
        while(j <= m):
            count += 1
            error1 += (k*X[j] + b - ts[j])**2
            j += 1
        if error1 != 0:
            error1/count
        sum_error = sum_error + error1
    return sum_error

def calculate_vertical_error(x1,y1,ts,X):
    sum_error = 0
    for i in range(len(y1)-1):
        k = (y1[i+1] - y1[i])/(x1[i+1] - x1[i])
        b = y1[i] - k*x1[i]
        #得到插值直线的斜率和截距 即：y = kx + b
        j = ts.tolist().index(y1[i])
        m = ts.tolist().index(y1[i+1])
        while(j <= m):
            sum_error = sum_error + abs(k*X[j] + b - ts[j])
            j += 1
    return sum_error
        
#----------------------标准化数据---------------------
def norm_data(ts):
   mean_ts = np.mean(ts) 
   std_ts = np.std(ts)
   ts = (ts - mean_ts) / std_ts
   return ts
   