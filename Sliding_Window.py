# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:13:25 2020

@author: Administrator
The generic Sliding Window algorithm
 
Algorithm Seg_TS = Sliding_Window(T, max_error)
anchor = 1;
while not finished segmenting time series
i = 2;
    while caculate_error(T[anchor: anchor + i]) < max_error
        i = i + 1;
    end;
    Seg_TS = concat(Seg_TS,create_segment(T[anchor: anchor+(i-1)]));
    anchor = anchor + i;
end;
"""
#定义全局最大误差
import pandas as pd
import numpy as np
import methods as md
import matplotlib.pyplot as plt
import time
max_error = 0.4

#-----------滑动窗口算法---------------

#加载数据函数
def loadDataSet(fileName):
    dataSetList = []
    df = pd.read_csv(fileName)
    #dataSetList = df.drop(['Date'], axis = 1)
    dataSetList = df
    return dataSetList


def Sliding_Window(T, max_error,X):
    Seg_TS = []
    anchor = 0
    while(anchor < len(T)-1 ):#分段结束条件，锚点到达最后一个点
        i = 1  #定义初始分段相对于锚点anchor的结束点
        while calculate_error(X[anchor:anchor + i + 1],T[anchor:anchor + i + 1]) < max_error:
            i = i + 1
            if anchor+i-1==len(T):
                break
        Seg_TS.append(T[anchor:anchor+i-1])
        anchor = anchor + i -1
    return Seg_TS
        
#-----------------------计算拟合误差---------------------
def calculate_error(X,Ts):
    #1.获取拟合直线的方程y = a*x + b----这里自变量的量纲问题如何处理，用Ts下的坐标岂不是忽略了它本身的时间序列坐标
    #Ts = Ts.tolist()
    #X = X.tolist()
    #a = abs((Ts[-len(Ts)]-Ts[-1])/len(Ts))
    #b = Ts[-1] - a
    a,b = md.min_error_func(X,Ts)
    sum_error = 0
    for i in range(len(Ts)):
        sum_error += (Ts[i] - (a*X[i] + b))**2
    sum_error = sum_error/len(Ts)
    return sum_error#返回垂直方向的均方误差

if __name__=='__main__':
    start =time.clock()
    TS = md.loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test2.csv')
    TS_x = md.loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test2.csv')
    ts = np.array(TS_x['Close'])
    ts = ts.tolist()
    ts_x = np.array(TS_x['Date'])
    TS_x = np.array(TS_x['Date'])
    TS_x = md.time_translate(TS_x)
    std_TS_x = np.std(TS_x)#时间序列的原始数据方差
    mean_TS_x = np.mean(TS_x)#时间序列的原始数据均值
    x = (TS_x - mean_TS_x)/std_TS_x #将时间戳x做标准化处理，便于传参进行后续计算
    TS_1 = np.array(TS['Close'])
    std_TS_1 = np.std(TS_1)#收盘价的原始数据方法
    mean_TS_1 = np.mean(TS_1)#收盘价的原始数据均值
    y = (TS_1 - mean_TS_1)/std_TS_1
    ans_T=[]
    ans_T=Sliding_Window(y,max_error,x)
    #取出ans_T中的分段点
    ans_seg_point=[]
    for i in range(len(ans_T)):
        ans_seg_point.append(ans_T[i][0])
        ans_seg_point.append(ans_T[i][-1])
    #将归一化后的数据还原
    ans_seg_point_date=[]
    y1 = []
    x1 = []
    for i in range(len(ans_seg_point)):
        y1.append(ans_seg_point[i])
    ans_seg_point_date1=[]
    for i in range(len(ans_seg_point)):
        ans_seg_point[i]=ans_seg_point[i] * std_TS_1 + mean_TS_1
        ans_seg_point_date.append(ts_x[ts.index(ans_seg_point[i])])
        x1.append(x[ts.index(ans_seg_point[i])])
    plt.figure(figsize=(12,6))
    plt.plot(ts_x,ts,'r--',label='raw_data')
    plt.scatter(ans_seg_point_date,ans_seg_point,color='blue',label='seg_point')
    plt.plot(ans_seg_point_date,ans_seg_point,color='green',label='fitting_line')
    plt.legend()
    plt.xticks(ts_x, ts_x, rotation=90, fontsize=10)
    plt.show()
    #----------------------计算拟合误差---------------------
    #用插值直线与原数据的均方差来表示error
    #参数描述： x
    
    #在原始序列中找到ans_seg_point_date 数组的标准值
    #ans_seg_point_date1 = md.norm_data(ans_seg_point_date1)
    #ans_seg_point = md.norm_data(ans_seg_point)
    error = md.calculate_fitting_error(x1,y1,y,x)
    error2 = md.calculate_vertical_error(x1,y1,y,x)
    print("the sliding window fitting error1: ",error)
    print("the sliding window fitting error2: ",error2)
    #T = loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
    #T = np.array(T['Close'])
    #ans_T = []
    #ans_T = Sliding_Window(T,max_error)
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))
