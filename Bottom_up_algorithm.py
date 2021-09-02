# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:37:15 2020

@author: Administrator

The bottom-up algorithm
-------------------------算法描述-----------------------
自底向上算法是自顶向下算法的自然补充，该算法先创建该时间序列的最佳近似，因此，n/2分段被用来近似的表示长度为n的时间序列
然后，计算合并每对相邻段的代价，该算法开始迭代的合并最低成本对，知道达到停止条件，当相邻的i和i+1段被合并，该算法需要做一些记录
首先，必须计算将新段与其右边分段合并的代价，此外，合并i-1段及其新的更大的临近分段的代价必须被重新计算

-------------------------算法伪代码----------------------
Algorithm Seg_TS = Bottom_Up(T, max_error)
for i = 1: 2 : length(T)                     #构造初始最佳逼近
    Seg_TS = concat(Seg_TS, create_segment(T[i: i+1]));
for i = 1: length(Seg_TS) - 1                #计算每一对合并代价
    merge_cost(i) = calculate_error((merge(Seg_TS[i], Seg_TS[i+1])))
while min(merge_cost) < max_error:
    i = min(merge_cost)                      #找到最低的合并代价去合并
    Seg_TS[i] = merge(Seg_TS[i], Seg_TS[i+1])#合并操作
    delete(Seg_TS[i+1])                      #合并后删除该分段
    merge_cost[i] = calculate_error(merge(Seg_TS[i], Seg_TS[i+1]))
    merge_cost[i-1] = calculate_error(merge(Seg_TS[i-1], Seg_TS[i]))
return Seg_TS

"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  ##绘图库
from scipy.optimize import leastsq  ##引入最小二乘法算法
max_error=0.9

#-----------------------时间戳转化为日期数组函数------------------------
def timeStamp_to_date(Xi):
    t1 = []
    otherStyleTime = []
    for i in range(len(Xi)):   
        t1.append(time.localtime(Xi[i]))
        otherStyleTime.append(time.strftime('%Y/%m/%d',t1[i]))
    out = np.array(otherStyleTime)
    return out
    
#-----------------------日期戳转化为时间戳数组函数------------------------
def time_translate(Xi):
    #a1 = "2019-5-10 23:40:00"
    # 先转换为时间数组
    #timeArray = time.strptime(Xi, "%Y-%m-%d %H:%M:%S")
    timeArray = []  
    timeStamp = []
    for i in range(len(TS_x)):
        timeArray.append(time.strptime(Xi[i], "%Y/%m/%d"))
        # 转换为时间戳
        timeStamp.append(int(time.mktime(timeArray[i])))
    print(timeStamp)
    return np.array(timeStamp)
     

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
    print("k=",k,"b=",b)
    print("cost："+str(Para[1]))
    print("求解的拟合直线为:")
    print("y="+str(round(k,2))+"x+"+str(round(b,2)))

    '''
   绘图，看拟合效果.
   matplotlib默认不支持中文，label设置中文的话需要另行设置
   如果报错，改成英文就可以
   '''

    #画样本点
   # plt.figure(figsize=(8,6)) ##指定图像比例： 8：6
    plt.figure(figsize=(8,6)) ##指定图像比例： 8：6
    plt.scatter(Xi,Yi,color="green",label="demo_data",linewidth=2) 

    #画拟合直线
    x=np.linspace(-2,2,100) ##在0-15直接画100个连续点
    y=k*x+b ##函数式
    plt.plot(x,y,color="red",label="approximate_line",linewidth=2) 
    plt.legend(loc='lower right') #绘制图例
    plt.show()
    
    return k,b#返回直线的系数k和b

#-------------------------加载数据函数----------------------
def loadDataSet(fileName):
    dataSetList = []
    df = pd.read_csv(fileName)
    #dataSetList = df.drop(['Date'], axis = 1)
    dataSetList = df
    return dataSetList

#-----------------------计算合并代价算法代码---------------------
def calculate_error(X,Ts):
    #1.获取拟合直线的方程y = a*x + b----这里自变量的量纲问题如何处理，用Ts下的坐标岂不是忽略了它本身的时间序列坐标
    #Ts = Ts.tolist()
    #X = X.tolist()
    #a = abs((Ts[-len(Ts)]-Ts[-1])/len(Ts))
    #b = Ts[-1] - a
    a,b = min_error_func(X,Ts)
    sum_error = 0
    for i in range(len(Ts)):
        sum_error += (Ts[i] - (a*X[i] + b))**2
    sum_error = sum_error/len(Ts)
    return sum_error#返回垂直方向误差

#-----------------------合并两个分段---------------------
def merge(Ts_1,Ts_2):
    list1 = Ts_1.tolist()   #转化为list便于合并操作
    list2 = Ts_2.tolist()
    for i in range(len(list2)):
        list1.append(list2[i])
    Ts = np.array(list1)
    return Ts
    

#-------------------------自底向上算法代码----------------------
def Bottom_Up(X,T, max_error):
    Seg_TS = []
    i = 0
    while(i<len(T)):
        Seg_TS.append(T[i:i+2]) #构造初始最佳逼近，以步长为2，构造分段
        i = i + 2
    merge_cost = []                      #用来存放合并代价
    for i in range(len(Seg_TS)-1):         #计算每一对合并代价
        merge_Seg_TS = merge(Seg_TS[i],Seg_TS[i+1])  #合并相邻的分段
        merge_cost.append(calculate_error(X[T.tolist().index(merge_Seg_TS[0]):(T.tolist().index(merge_Seg_TS[-1])+1)],merge_Seg_TS))#如何计算合并代价----->合并后的分段误差
    while min(merge_cost) < max_error:
        i = merge_cost.index(min(merge_cost))  #找到最低的合并代价去合并
        if i < len(Seg_TS)-1:#如果最小代价的分段不是Seg_TS的最后一个分段
            Seg_TS[i] = merge(Seg_TS[i], Seg_TS[i+1])#合并操作
            Seg_TS.pop(i+1)                  #合并后删除
        else:
            break
        if i == len(Seg_TS)-1:#这时候是Seg_TS的最后一个分段和倒数第二个分段合并了，此时，只需更新i-1和i的合并误差
            merge_0 = merge(Seg_TS[i-1], Seg_TS[i])
            merge_cost[i-1] = calculate_error(X[T.tolist().index(merge_0[0]):(T.tolist().index(merge_0[-1])+1)],merge_0)
        else:#更新合并操作后分段后的误差
            merge_1 = merge(Seg_TS[i], Seg_TS[i+1])
            merge_2 = merge(Seg_TS[i-1], Seg_TS[i])
            merge_cost[i] = calculate_error(X[T.tolist().index(merge_1[0]):(T.tolist().index(merge_1[-1])+1)],merge_1)
            merge_cost[i-1] = calculate_error(X[T.tolist().index(merge_2[0]):(T.tolist().index(merge_2[-1])+1)],merge_2)
    return Seg_TS

if __name__=='__main__':
    TS = loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
    TS_x = loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
    TS_x = np.array(TS_x['Date'])
    TS_x = time_translate(TS_x)
    std_TS_x = np.std(TS_x)#时间序列的原始数据方差
    mean_TS_x = np.mean(TS_x)#时间序列的原始数据均值
    x = (TS_x-mean_TS_x)/std_TS_x
    TS = np.array(TS['Close'])
    std_TS = np.std(TS)#收盘价的原始数据方法
    mean_TS = np.mean(TS)#收盘价的原始数据均值
    y = (TS - mean_TS)/std_TS
    Seg_TS = Bottom_Up(x,y, max_error)
    #将标准化后的数据还原到到原来的结果
    list1 = []#用来存入最后的结果
    for i in range(len(Seg_TS)):
        list1.append(Seg_TS[i].tolist())
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            list1[i][j] = list1[i][j]*std_TS + mean_TS
    for i in range(len(list1)):
        list1[i] = np.array(list1[i])
    Seg_TS = np.array(list1)
    
#----------------代码分析----------------------
"""
1.总体逻辑没有问题，代码可以run起来
问题如下:
    1.本页的第15行，构造最佳初始逼近分段，这里的构造具体方法不太了解，我这里用的是以步长为2，从第一个点开始分段，每两个点构成一个分段不知道理解是否有误差
    2.本页的17,18行，计算每一对的代价的方法选取不是最佳，用的是第一个点和最后一个点模拟的直线，用此直线和每一个点的垂直(SVD)距离来当做误差，导致误差太大
"""
        
        
        
    
    
        
