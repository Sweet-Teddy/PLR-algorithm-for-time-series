# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:20:38 2020

@author: Administrator
"""
#---------------------算法 4-1 伪代码----------------------
'''
输入：整体时间序列TS流入第i个SW; SW的长度m;
输出：转折点TPs的权重优先级队列PrioL（按照WTP从大到小的顺序）

1: tpF = TS.get(i*m)  #前一个TP，根据定义4.2,为了计算第1个TP的权重，将SWi中的的起始时序点设置为默认值
2: tpC = null; #待计算权重的TP
3: j = 1;
4: while j<m do:
5:  vt = TS.get(i*m + j)
6: CurTS.add(vt); #当前时间序列SWi的存储列表CurTS
7: if vt is a TP then:
    if tpC != null then:
        tpC.weight = calcWTP(tpF,tpC,vt) #根据定义4.2计算tpC的权重WTP
        sortByWTP(PrioL,tpC) #按照WTP从大到小顺序在PrioL中排序
        tpF = tpC
        tpC = vt;
    else:
        tpC = vt
    end if
   end if
   j ++
  end while
  vt = curTS.get(m-1) #获取当前时间序列SWi中终止时序点
  tpC.weight = calcWTP(tpF,tpC,vt) #根据定义4.2，利用终止时序点计算，最后一个TP的权重WTP
  sortByWTP(PrioL,tpC) #按照WTP从大到小的顺序在PrioL中进行排序
  Head = creatARList(curTS,PrioL) #根据当前TPs数目创建Head节点并向后开辟空间
  return PrioL
  
MPLR-SBT中重要感知点PIPs与重要点的定义基本相同，参考文献55
'''
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt  ##绘图库
from scipy.optimize import leastsq  ##引入最小二乘法算法



#-------------------定义ListNode类实例的基本属性说明-------------------
class ListNode:
    tag = -1 #当前节点如果不是Head节点，默认值为-1
    index = None  #当前转折点丁P在原始时间序列中的时刻值
    value = None  #当前转折点TP的数据值
    rank = None   #当前转折点TP的权重值排名(按照权重值从大到小的顺序)
    es_L = None   #将当前转折点T尸作为分段终止点(P，)的分段误差
    es_R = None   #将当前转折点丁尸作为分段起始点(Pb)的分段误差
    mes = None    #利用前rank个转折点TPs进行线性分段表示时，当前的最大分段误差
    ets = None    #利用前rank个转折点TPs进行线性分段表示时，当前的整体分段误差
    parL = None   #对原始时间序列进行(rank+1)维的PAA操作，获得的分段聚合近似结果
    



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

#-----------------------加载数据函数--------------------
def loadDataSet(fileName):
    dataSetList = []
    df = pd.read_csv(fileName)
    #dataSetList = df.drop(['Date'], axis = 1)
    dataSetList = df
    return dataSetList

#拟合直线line(va，vc)与真实时间序列{Va,...Vb,...,Vc}的单点误差累加值，作为转折点Vb的权重WTP
#-------------------函数参数说明--------------
#Ts, 当前滑动窗口 起点为tpF,终点为tpC  ,vt为需要计算权重的当前SW中的时序点
def calcWTP(Ts,X,tpF, tpC, vt):
    sum_error = 0
    k = (tpF-tpC)/(X[Ts.index(tpF) - X[Ts.index(tpC)]]) #计算由tpC 和 tpF构成拟合直线的斜率k
    b = tpF - k*X[Ts.index(tpF)] #截距b
    for i in range(len(Ts)):
        sum_error += abs((k*X[i] + b) - Ts[i]) #计算累计误差
    return sum_error
    
#-----------------按照WTP排序函数------------------
def sortByWTP(PrioL, tpC):
    for i in range(len(PrioL) - 1):
        if tpC.weight <= PrioL[i].weight and tpC.weight >= PrioL[i+1]:
            PrioL.insert(   i + 1,tpC)
    return PrioL

def creatARList(curTS, PrioL):
    
    
    
#---------------函数参数说明-----------------
#TS 整体时间序列   X为时间戳    Tp为转折点序列  i 表示第i个SW    m：  sw的长度
def Algorithm_4_1(TS, X, Tp, i, m):
    PrioL = [] #优先级队列
    curTS = [] #当前时间序列SWi的存储列表CurTS
    #定义ListNode类Head实例
    Head = ListNode()
    tpF = TS[i*m - 1] #tfF为起始时序点
    tpC = None  #待计算权重的TP
    j = 1
    while j < m :
        vt = TS[i*m + j -1]
        curTS.append(vt)    #当前时间序列SWi的存储列表CurTS
        if vt in Tp: #如果点vt是转折点
            if  tpC != None :
                tpC.weight =  calcWTP(TS[TS.index(tpF) : TS.index(tpC) + 1], X[TS.index(tpF) : TS.index(tpC) + 1],tpF,tpC,vt) #计算tpC的权重WTP
                PrioL = sortByWTP(PrioL, tpC) #按照WTP从大到小顺序在PrioL（优先级队列）中排序
                tpF = tpC
                tpC = vt
            else:
                tpC = vt
        j += 1
    vt = curTS[m-1] #获取当前时间序列SWi中终止时序点
    tpC.weight = calcWTP(TS[TS.index(tpF) : TS.index(tpC) + 1], X[TS.index(tpF) : TS.index(tpC) + 1],tpF, tpC, vt)#利用终止时序点计算，最后一个TP的权重WTP
    sortByWTP(PrioL, tpC) #按照WTP从大到小的顺序在PrioL中进行排序
    Head = creatARList(curTS,PrioL) #根据TPs数目创建Head节点并向后开辟空间
    return PrioL #转折点TPs的权重优先级队列PrioL（按照WTP从大到小的顺序）

if __name__=='__main__':
    TS = loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
    TS_x = loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
    ts = np.array(TS_x['Close'])
    ts = ts.tolist()
    TS_x = np.array(TS_x['Date'])
    TS_x = time_translate(TS_x)
    std_TS_x = np.std(TS_x)#时间序列的原始数据方差
    mean_TS_x = np.mean(TS_x)#时间序列的原始数据均值
    x = (TS_x - mean_TS_x)/std_TS_x #将时间戳x做标准化处理，便于传参进行后续计算
    TS_1 = np.array(TS['Close'])
    std_TS_1 = np.std(TS_1)#收盘价的原始数据方法
    mean_TS_1 = np.mean(TS_1)#收盘价的原始数据均值
    y = (TS_1 - mean_TS_1)/std_TS_1
    #-------------------获取转折点Tp-------------
    #Tp=[.............]
    Tp = []
    #i 为第 i 个滑动窗口，m为滑动窗口的长度
    Algorithm_4_1(TS,x,Tp,i,m)

