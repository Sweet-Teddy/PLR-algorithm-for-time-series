# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:24:01 2020

@author: Administrator

PLR实现
算法步骤：
输入：时间序列TS；预设滑动窗口中最大线性分段数num;预设最大单点误差MEP（）
输出： 线性分段点队列,segList

1.curnum = 0 //当前线性分段数初始化
2.sps = vti //设置当前分段起始点
3.tpIList.push(vti) //将分段起始点设置为重要转折点
4.segList.push(sps)  //线性分段点进队
5.Segsup = 0 //当前分段斜率上界初始化为0 
6.Seglow = 0 //当前分段斜率下界初始化为0
7.while curnum <= num  do
8.   while Segsup >= Seglow do
9.       i++
         if  Vti is  a TPB then  //如果此点是基本转折点，将其进栈，再判断是否为重要转折点
             tpBList.push(Vti)
             TPI = tpIList.pop() //取出与Vti最近的且在其左侧的重要转折点，用来判断Vti是否为重要转折点
             if  Vti is a TPI compared to TPI then:
             tpIList.push(Vti);tpIList.push(Vti);
             end if
         end if
         update Segsup(sps,Vti) //更新当前分段斜率上确界
         update Seglow(sps,Vti) //更新当前分段斜率下确界
     end while
     sps = Vti //设置当前分段终止点以及下一次分段的起始点
     segList.push(sps)
     curnum ++
  end while
  return segList
         
"""
#数据用股票的话，取收盘价Close作为参考标准
import numpy as np
import methods as md
import matplotlib.pyplot as plt
#p=MEPP×10％。  μ=SWlen×10％，

#MEPP在本实验中一般取值为：{10％，20％，30％，⋯，}

#MEPP取0.1  ρ=0.01 μ=滑动窗口长度为多少
ρ = 0.01
μ = 0.6 #定义两个全局变量用来判断是否为重要基本点
MEP=0.3#定义分段误差
seg_num=1000 #定义分段数 
#------------------参数说明--------------------------
# TS,标准化后的时间序列   X 标准化后的自变量  num分段数  MEP分段误差
def segmentation(TS,X, num, MEP):
    curnum = 0
    tpIList = []
    tpBList = []
    SegList = []
    i = 0 #初始化TS数据的第一个点下标
    sps = TS[i] #将第一个点作为线性分段起始点
    sps_index = 0
    tpIList.append(TS[0]) #线性分段点入栈
    tpBList.append(TS[0])
    SegList.append(sps)
    Segsup = 9999
    Seglow = -9999 #初始化当先分段斜率上下界为0
    
    def update_Segsup(TS,X, i,j):
        """Segsup的整体斜率上界和下界由如下关系确定
        Segsup(Vti, Vtj) = min sup(Vti, Vtt)
        Seglow(Vti, Vtj) = max low(Vti, Vtt)"""
        min_sup = 9999
        k = i
        while(k < j + 1 ):
            k = i + 1
            #print('更新了一次上界')
            if(((TS[k] + MEP - TS[i]) / (X[k]-X[i])) <= min_sup): min_sup = (TS[k] + MEP - TS[i]) / (X[k]-X[i])#判断当前点与分段起始点构成直线的斜率是否比当前分段最大的斜率大
            return min_sup
        
    def update_Seglow(TS,X, i, j):
        max_low = -9999
        k = i
        while(k < j + 1):
            k = i + 1
            #print('更新了一次下界')
            if(((TS[k]-MEP - TS[i]) / (X[k]-X[i])) >= max_low): max_low = (TS[k] - MEP - TS[i]) / (X[k]-X[i])##判断当前点与分段起始点构成直线的斜率是否比当前分段最小的斜率小
            return max_low
            
        
            
    while curnum <= num:
        Segsup = 9999
        Seglow = -9999 #初始化当先分段斜率上下界为0
        while Segsup >= Seglow and i < len(TS) - 1:
            i += 1
            #print('i=',i)
            if(0<i<len(TS)-1):
                if((TS[i] > TS[i-1] and TS[i] > TS[i+1]) or (TS[i] < TS[i-1] and TS[i] < TS[i+1]) or (TS[i] == TS[i-1] and TS[i] > TS[i+1]) 
                or (TS[i] == TS[i-1] and TS[i] < TS[i+1]) or (TS[i] > TS[i-1] and TS[i] == TS[i+1]) or (TS[i] < TS[i-1] and TS[i] == TS[i+1])):
                    #如果此点是基本转折点，将其进栈，再判断是否为重要转折点
                    tpBList.append(TS[i])
                    #print('转折点i=',i)
                    tpI = tpIList.pop() #取出与TS[i]最近的且在其左侧的重要转折点，用来判断TS[i]是否为重要转折点，判断如下：
                    if(abs(TS[i] - tpI) / ((abs(TS[i]) + abs(tpI)) / 2) >= ρ):
                        tpIList.append(tpI),tpIList.append(TS[i]) #将出栈的tpI先入栈，再入栈是重要转折点的TS[i]
                    else: #如果当前点不是重要转折点，就将tpI加入到tpIList中，然后将其置为空
                        tpIList.append(tpI)
                        tpI = None
            #Segsup = update_Segsup(TS,X, sps_index ,i)
            Segsup = min(Segsup,((TS[sps_index]-(TS[i] + MEP))/(X[sps_index] - X[i])))
            Seglow = max(Seglow,((TS[sps_index]-(TS[i] - MEP))/(X[sps_index] - X[i])))
            #Seglow = update_Seglow(TS,X, sps_index ,i)#更新上界和下界
        sps_index = i
        sps = TS[i]
        if SegList[-1] != sps:
            SegList.append(sps)
        curnum += 1 #当前分段完毕，进行下一步分段
        #print('当前分段完毕，进行下一步分段!')
    #将原始时间序列的最后一个点加入到分段列表中
    #SegList.append(y[-1])
    return SegList,tpIList,tpBList



if __name__ == '__main__':
    TS = md.loadDataSet(r'E:\论文\论文初写\data\test.csv',encoding='utf-8')
    TS_x = md.loadDataSet(r'E:\论文\论文初写\data\test.csv',encoding='utf-8')
    ts = np.array(TS_x['Close'])
    ts = ts.tolist()
    ts_x = np.array(TS_x['Date'])
    TS_x = np.array(TS_x['Date'])
    TS_x = md.time_translate(TS_x)
    std_TS_x = np.std(TS_x)#时间序列的原始数据方差
    mean_TS_x = np.mean(TS_x)#时间序列的原始数据均值
    X = (TS_x - mean_TS_x)/std_TS_x #将时间戳x做标准化处理，便于传参进行后续计算
    TS_1 = np.array(TS['Close'])
    std_TS_1 = np.std(TS_1)#收盘价的原始数据方法
    mean_TS_1 = np.mean(TS_1)#收盘价的原始数据均值
    y = (TS_1 - mean_TS_1)/std_TS_1
    segList,tpIList,tpBList = segmentation(y,X,seg_num,MEP)#TS为传入数据集，2为自定义分段数，0.1为分段误差MEP
    x1 = []
    y1=[]
    for i in range(len(segList)):
      y1.append(segList[i]) 
      x1.append(X[y.tolist().index(y1[i])])
# =============================================================================
#     #原始时间序列复原
#     for i in range(len(segList)):
#         segList[i] = segList[i] * std_TS_1 + mean_TS_1
#     for i in range(len(tpIList)):
#         tpIList[i] = tpIList[i] * std_TS_1 + mean_TS_1
#     for i in range(len(tpBList)):
#         tpBList[i] = tpBList[i] * std_TS_1 + mean_TS_1
#     ans_seg_point_date=[]
#     for i in range(len(segList)):
#         ans_seg_point_date.append(ts_x[ts.index(segList[i])])
# =============================================================================
# =============================================================================
#     plt.figure(figsize=(12,6))
#     plt.plot(ts_x,ts,'r--',label='raw_data') 
#     plt.scatter(ans_seg_point_date,segList,color='blue',label='seg_point')
#     plt.plot(ans_seg_point_date,segList,color='green',label='fitting_line')
#     plt.legend()
#     plt.xticks(ts_x, ts_x, rotation=90, fontsize=10)
#     plt.show() 
#     error = md.calculate_fitting_error(x1,y1,y,X)
#     error2 = md.calculate_vertical_error(x1,y1,y,X)
#     print("the sliding window mean square fitting error1: ",error)
#     print("the sliding window vertical fitting error2: ",error2)
# =============================================================================
