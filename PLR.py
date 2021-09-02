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
import pandas as pd
import numpy as np
ρ = 122
μ = 0.1 #定义两个全局变量用来判断是否为重要基本点
MEP=0.1#定义分段误差

def loadDataSet(fileName):
    dataSetList = []
    df = pd.read_csv(fileName)
    #dataSetList = df.drop(['Date'], axis = 1)
    dataSetList = df
    return dataSetList
    
def segmentation(TS, num, MEP):
    curnum = 0
    tpIList = []
    tpBList = []
    SegList = []
    i = 0 #初始化TS数据的第一个点下标
    sps = TS[i] #将第一个点作为线性分段起始点
    sps_index = 0
    tpIList.append(TS[0]) #线性分段点入栈
    SegList.append(sps)
    Segsup = 0
    Seglow = 0 #初始化当先分段斜率上下界为0
    
    def update_Segsup(TS, i,j):
        """Segsup的整体斜率上界和下界由如下关系确定
        Segsup(Vti, Vtj) = min sup(Vti, Vtt)
        Seglow(Vti, Vtj) = max low(Vti, Vtt)"""
        min_sup = -9999
        k = i
        while(k < j + 1 ):
            k = i + 1
            print('更新了一次上界')
            if(((TS[k] + MEP - TS[i]) / (k-i)) >= min_sup): min_sup = abs(TS[k] + MEP - TS[i]) / abs(k-i)#判断当前点与分段起始点构成直线的斜率是否比当前分段最大的斜率大
            return min_sup
        
    def update_Seglow(TS, i, j):
        max_low = 9999
        k = i
        while(k < j + 1):
            k = i + 1
            print('更新了一次下界')
            if(((TS[k]-MEP - TS[i]) / (k-i)) <= max_low): max_low = abs(TS[k] - MEP - TS[i]) / abs(k-i)##判断当前点与分段起始点构成直线的斜率是否比当前分段最小的斜率小
            return max_low
            
        
            
    while curnum <= num:
        while Segsup >= Seglow:
            i += 1
            print('i=',i)
            if(0<i<len(TS)-1):
                if((TS[i] > TS[i-1] and TS[i] > TS[i+1]) or (TS[i] < TS[i-1] and TS[i] < TS[i+1]) or (TS[i] == TS[i-1] and TS[i] > TS[i+1]) 
                or (TS[i] == TS[i-1] and TS[i] < TS[i+1]) or (TS[i] > TS[i-1] and TS[i] == TS[i+1]) or (TS[i] < TS[i-1] and TS[i] == TS[i+1])):
                    #如果此点是基本转折点，将其进栈，再判断是否为重要转折点
                    tpBList.append(TS[i])
                    #print('转折点i=',i)
                    tpI = tpIList.pop() #取出与TS[i]最近的且在其左侧的重要转折点，用来判断TS[i]是否为重要转折点，判断如下：
                    if(abs(TS[i] - tpI) / ((abs(TS[i]) + abs(tpI)) / 2) >= ρ):
                        tpIList.append(tpI),tpIList.append(TS[i]) #将出栈的tpI先入栈，再入栈是重要转折点的TS[i]
            Segsup = update_Segsup(TS, sps_index ,i)
            Seglow = update_Seglow(TS, sps_index ,i)#更新上界和下界
        sps_index = i
        sps = TS[i]
        SegList.append(sps)
        curnum += 1 #当前分段完毕，进行下一步分段
        #print('当前分段完毕，进行下一步分段!')
    return SegList,tpIList,tpBList

if __name__ == '__main__':
    TS = loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
    TS = np.array(TS['Close'])
    SegList,tpIList,tpBList = segmentation(TS,4,0.1)#TS为传入数据集，2为自定义分段数，0.1为分段误差MEP
    