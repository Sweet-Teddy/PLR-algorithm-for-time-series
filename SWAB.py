# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:22:22 2020

@author: Administrator

The SWAB(Sliding Window and Bottom-up) algorithm
-------------------------算法描述-----------------------
鉴于滑动窗口算法的主要问题是它无法向前看，缺乏离线(批量)副本的全局视图。自底向上和自顶向下的方法产生更好的结果，但是是离线的，需要扫描整个数据集
因此，我们引入了一种新颖的方法，在这种方法中，我们捕获了滑动窗口的在线特性，同时又保留了自底向上的优势

-------------------------算法伪代码-----------------------
Algorithm Seg_TS = SWAB(max_error, seg_num) #分段数为整数，大概为5或6

read in data points to fill w      #w是缓冲区，足够用来逼近分段数？------------>缓冲区如何确定？
    lower_bound = (size of w) / 2  #上限(和下限)是初始缓冲区的两倍(和一半)。
    upper_bound = 2 * (size of w)  #下限是初始缓冲区的两倍

while data at input:
    T = Bottom_Up(w, max_error)    #调用传统的自底向上算法
    Seg_TS = CONCAT(SEG_TS, T(1))  #向右滑动窗口
    w = TAKEOUT(w, w')             #从缓冲区w删除T(1)中的w'点
    if data at input               #从BEST_LINE()添加点到缓存区w中
        w = CONCAT(w, BEST_LINE(max_error)) #检查上界和下届，如果需要作出调整
    else:
        #从缓存区更新近似分段
        Seg_TS = CONCAT(SEG_TS, (T-T(1)))
    end
end

Function S = BEST_LINE(max_error)   #返回S点
    while error<= max_error         下一个潜在的分段
        read in one additional data point, d, into S
        S = CONCAT(S, d)
        error = approx_segment(S)
    end while
    return s

"""

import numpy as np
import pandas as pd
import Bottom_up_algorithm as Bottom_Up

#-------------------------加载数据函数-----------------------
def loadDataSet(fileName):
    dataSetList = []
    df = pd.read_csv(fileName)
    #dataSetList = df.drop(['Date'], axis = 1)
    dataSetList = df
    return dataSetList

#-------------------------SWAB算法-----------------------
def SWAB(TS,X,max_error, seg_num):
    lower_bound = TS[0].length / 2
    upper_bound = 2 * TS[0].length
    i = 0
    T = []
    while i < TS.length:
        T = Bottom_Up(X[i],TS[i],max_error)
        
    
    
    

