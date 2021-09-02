# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:38:21 2020

@author: Administrator

The generic top-down algorithm

-------------------------算法描述-----------------------
自顶向下算法考虑了时间序列的每一个可能的分段，并将其划分到最佳的位置，思想有点像二分法
然后看这些分段的近似（分段）（**）误差是否低于预设的error，不满足误差要求，就递归的分割子序列
直到所有的子序列的最大误差都低于阈值，算法如下

-------------------------算法伪代码----------------------
Algorithm Seg_TS = Top_Down(T, max_error) #传入时间序列T和预设的最大误差

best_so_far = inf;
for i = 2 to length(T) - 2 #找到最佳的位置去分割
    improvement_in_approximation = improvement_spliting_here(T,i)
    if improvement_in_approximation < best_so_far
        breakpoint = i;
        best_so_far = improvement_in_approximation;
    end;
end;
#如果需要，递归的分割左分段
if calculate_error(T[1:breakpoint]) > max_error
    Seg_TS = Top_Down(T[1:breakpoint]);
end;

#如果需要，递归的分割右分段
if calculate_error(T[breakpoint + 1 : length(T)]) > max_error
    Seg_TS = Top_Down(T[breakpoint + 1 : length(T)]);
end;

"""
import numpy as np
import pandas as pd
import methods as md
import time
#-------------------------问题----------------------
#def calculate_error(Ts):
    #这里误差如何计算，用的是什么标准
#def improvement_spliting_here(T,i):
    #此函数表达的是什么意思

#定义全局变量max_error
max_error = 0.99

#-------------------------计算误差函数----------------------
def calculate_error(Ts,X):
    #1.获取拟合直线的方程y = a*x + b----这里自变量的量纲问题如何处理，用Ts下的坐标岂不是忽略了它本身的时间序列坐标
    # 解决：用时间序列的时间戳，标准化后的去量纲数据得以解决
    #a = abs((Ts[-len(Ts)]-Ts[-1])/len(Ts))
    a = (Ts[-len(Ts)] - Ts[-1]) / (X[-len(Ts)] - Ts[-1])
    #b = Ts[-1] - a
    b = Ts[-1] - a * X[-1]
    sum_error = 0
    for i in range(len(Ts)):
        sum_error += (Ts[i] - (a*X[i] + b))**2
    sum_error = sum_error/len(Ts)
    return sum_error#返回垂直方向误差

#-----------------函数说明---------------
#先给定一个时间序列，然后确定从i=2开始的一个点，满足分段误差标准，然后递归的使每个子分段都满足这个要求
def improvement_spliting_here(T, i, X):
    #此函数表达的是什么意思
    if calculate_error(T[0:i],X[0:i]) < max_error and calculate_error(T[i: len(T)],X[i : len(T)]) < max_error:
        return i
    else:
        return False
    

def Top_Down(T, X, max_error):#函数的返回值要确定，是否就是最后分段后的时间序列
    best_so_far = 999 #预设
    i = 2
    seg_TS = []
    while(i < len(T) - 2):
        #此处improvement_spliting_here()函数表示应该是判断从点i处分割，是否满足从点i出分割，得到的分段满足误差要求
        improvement_in_approximation = improvement_spliting_here(T,i,X)
        if improvement_in_approximation < best_so_far:
            breakpoint = i
            best_so_far = improvement_in_approximation
    #计算误差，不满足继续递归分割左分段
    if calculate_error(T[0:breakpoint]) > max_error:
        seg_TS = Top_Down(T[0:breakpoint],X[0:breakpoint])
    if calculate_error(T[breakpoint : len(T)], X[breakpoint: len(T)]):
        seg_TS = Top_Down(T[breakpoint : len(T)],X[breakpoint : len(T)])
    return seg_TS

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
    print(timeStamp)
    return np.array(timeStamp)

if __name__ == '__main__':
    TS = md.loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
    TS_x = md.loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
    ts = np.array(TS_x['Close'])
    ts = ts.tolist()
    TS_x = np.array(TS_x['Date'])
    TS_x = time_translate(TS_x)
    std_TS_x = np.std(TS_x)#时间序列的原始数据方差
    mean_TS_x = np.mean(TS_x)#时间序列的原始数据均值
    X = (TS_x - mean_TS_x)/std_TS_x #将时间戳x做标准化处理，便于传参进行后续计算
    TS_1 = np.array(TS['Close'])
    std_TS_1 = np.std(TS_1)#收盘价的原始数据方法
    mean_TS_1 = np.mean(TS_1)#收盘价的原始数据均值
    y = (TS_1 - mean_TS_1)/std_TS_1
    segList_Top_Down = []
    segList_Top_Down = Top_Down(y, X, max_error)

