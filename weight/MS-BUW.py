# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:44:01 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:55:30 2020

@author: Administrator
"""
import methods as md
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
sys.setrecursionlimit(10090)




def getMedian(Ts,start,end,segList):
    #print("获取了一次median")
    n = end - start + 1 
    #start = Ts.tolist().index(start)
    #end = Ts.tolist().index(end)
    #第一步先将Ts排序，然后再选取中位数点
    Ts_sort = Ts.tolist()
    Ts_sort.sort()
    
    if (n % 2 == 1):#判断是否为奇数
        midnum = Ts_sort[int((start + end)/2)]
        if (midnum in segList and ((start + end)/2 + 1) < len(Ts_sort) - 1):
            midnum = Ts_sort[int((start + end)/2) + 1]
            if(midnum not in segList):
                return midnum
            else:
                for i in range(len(Ts_sort)):
                    if (Ts_sort[i] not in segList):
                        midnum = Ts_sort[i]
                        return midnum
        else:
            for i in range(len(Ts_sort)):
                    if (Ts_sort[i] not in segList):
                        midnum = Ts_sort[i]
                        return midnum
# =============================================================================
#         if(midnum not in segList):
#             return midnum
# =============================================================================
            
# =============================================================================
#             midnum = Ts_sort[int((start + end)/2) + 1]
#             return midnum
# =============================================================================
    else:
        midnum = Ts_sort[int((start + end)/2)]
        k = 0
        while(midnum in segList): #如果向下取整得到的分段点在分段点列表中
            k += 1
            midnum = Ts_sort[int((start + end)/2) + k] #选取midnum后面一个点
        return midnum
        
        
        
def calculateError(tss,Xx):
    #x = X[(ts.tolist().index(ts[0])) : (ts.tolist().index(ts[-1])) + 1] #找到与ts对应的时间序列x，便于后面分段误差的计算
    k = (tss[0] - tss[-1]) / (Xx[0] - Xx[-1])
    b = tss[0] - k * Xx[0]
    sum_error = 0
    for i in range(len(tss) - 1):
        sum_error += abs(k*Xx[i] + b - tss[i])
    return sum_error
    


def median_plr_1(TS,start,end,MES,segList,X):
    midnum = getMedian(TS, start, end,segList) #获取当前分段的中位数 
    if (midnum != None):
        segList.append(midnum)
    midnum_index = TS.tolist().index(midnum)
    #print(midnum_index)
    Ts_left = TS[start : midnum_index + 1]
    Ts_right = TS[midnum_index : end + 1] 
    
    if (len(Ts_left) <= 2):
        for i in range(len(Ts_left)):
            if(Ts_left[i] not in segList and Ts_left[i]!=None):
                segList.append(Ts_left[i])
        
    else:
        error_left = calculateError(Ts_left,X[y.tolist().index(TS[start]):y.tolist().index(TS[midnum_index])+1]) #左边分段的拟合误差
        if (error_left > MES and start < midnum_index): #如果左边分段拟合误差超过了分段误差
            segList.append(median_plr_1(Ts_left,start,midnum_index,MES,segList,x))
        
    if (len(Ts_right) <= 2):
        for i in range(len(Ts_right)):
            if(Ts_right[i] not in segList and Ts_right[i]!=None):
                segList.append(Ts_right[i])
         
    else:
        error_right = calculateError(Ts_right,X[y.tolist().index(TS[midnum_index]):y.tolist().index(TS[end])+1]) #右边分段的拟合误差
        if (error_right > MES and end > midnum_index): #如果右边分段拟合误差超过了分段误差
# =============================================================================
#             end = end - midnum_index
# =============================================================================
            segList.append(median_plr_1(Ts_right,start,end - midnum_index,MES,segList,x))
def mergeSeg_error(start, end, y, X):
    y_1 = y[y.tolist().index(start): (y.tolist().index(end) + 1)]
    x_1 = X[y.tolist().index(start) : (y.tolist().index(end) + 1)] #找到与ts对应的时间序列x，便于后面分段误差的计算
    k = (y_1[0] - y_1[-1]) / (x_1[0] - x_1[-1])
    b = y_1[0] - k * x_1[0]
    sum_error = 0
    for i in range(len(y_1)):
        sum_error += abs(y_1[i] - (k * x_1[i] + b))
    return sum_error



def mednum_merge(seg_1,MES_merge,y,X_meg):
    start = seg_1[0]
    i = 0
    seg_ans = []
    seg_ans.append(seg_1[0])
    end = seg_1[2]
    while(i < len(seg_1)-3):
        
        error_1 = mergeSeg_error(start, end,y, X_meg) #计算合并相邻两个分段后的分段误差
        if error_1 < MES_merge:
            #如果加入一个分段点后，满足merge_Error要求，则继续加入分段点
            end = seg_1[seg_1.index(end) + 1]
            
        else:
            
            seg_ans.append(end)
            start = end
            end = seg_1[seg_1.index(start) + 1]
            #end = seg_1[i + 2]
            #start = seg_1[i+1] #不满足合并条件则向后继续合并
        i += 1   
    seg_ans.append(seg_1[-1])
    return seg_ans
    
def plr_merge(seg_1,MES_merge,y,X_meg):
    start = seg_1[0]
    i = 0
    seg_ans = []
    while(i < len(seg_1)-2):
        end = seg_1[i + 2]
        error_1 = mergeSeg_error(start, end,y, X_meg) #计算合并相邻两个分段后的分段误差
        if error_1 < MES_merge:
            #如果加入一个分段点后，满足merge_Error要求，则合并当前分段
            seg_1[i + 1] = None
            start = end
            
        else:
            start = seg_1[i+1] #不满足合并条件则向后继续合并
        i += 2
    for val in seg_1:
        if val != None:
            seg_ans.append(val)
    return seg_ans    
    
if __name__=='__main__':
    start_time =time.clock()
    TS = md.loadDataSet(r'E:\论文\论文初写\data\CSI.csv',encoding='utf-8')
    #TS_x = md.loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\美国标准普尔500指数历史数据(1990-2019).csv')
    TS_x = md.loadDataSet(r'E:\论文\论文初写\data\CSI.csv',encoding='utf-8')
    ts = np.array(TS_x['Close'])
    ts_volume = np.array(TS_x['Volume'])
    ts = ts.tolist()
    ts_volume.tolist()
    ts_x = np.array(TS_x['Date'])
    TS_x = np.array(TS_x['Date'])
    TS_1 = np.array(TS['Close'])
    TS_volume = np.array(TS['Volume'])    
    for i in range(len(ts)):
        j = i + 1
        while(j<len(ts)):
            if(ts[i]==ts[j]):
                ts = np.delete(ts,j)
                ts_x = np.delete(ts_x,j)
                TS_x = np.delete(TS_x,j)
                TS_1 = np.delete(TS_1,j)
                TS_volume = np.delete(TS_volume,j)
            j += 1
    
    TS_x = md.time_translate(TS_x)
    std_TS_x = np.std(TS_x)#时间序列的原始数据方差
    mean_TS_x = np.mean(TS_x)#时间序列的原始数据均值
    x = (TS_x - mean_TS_x)/std_TS_x #将时间戳x做标准化处理，便于传参进行后续计算

    
    std_TS_volume = np.std(TS_volume)#交易量的原始数据方法
    mean_TS_volume = np.mean(TS_volume)#交易量的原始数据均值
    y_volume = (TS_volume - mean_TS_volume)/std_TS_volume
    
    std_TS_1 = np.std(TS_1)#收盘价的原始数据方法
    mean_TS_1 = np.mean(TS_1)#收盘价的原始数据均值
    y_close = (TS_1 - mean_TS_1)/std_TS_1
    
    y = 0.5 * y_volume + 0.5 * y_close
     #定义要存入文件中的结果列表
    list_error = [] #统计误差
    list_segNum=[] #统计分段数
    list_time = [] #运行时间
    list_ans = []
    
    MES = 0.1
    while(MES <=0.5):
        start_time =time.clock()
        MES_merge =5 * MES
        #----------------plr部分---------------
        start = 0
        end = len(y) - 1
        segList = [] #初始化分段点列表
        segList.append(y[start])
        segList.append(y[end]) #将列表起点和终点当做分段点加入列表中
        segList_MMP = []
        segList_MMP = median_plr_1(y,start,end,MES,segList,x)
        ses = []
        for val in segList:
            if val != None:
                ses.append(val)
        segList_MM = ses
        segList_meg = []
        #将分段点还原成原时间序列顺序
        for i in range(len(y)):
            if(y[i] in segList_MM):
                segList_meg.append(y[i])
    
        #-------------plr结束----------
        #segList_mergeMMP = mednum_merge(segList_meg, MES_merge,y, x)
        print("合并前分段长度：%d"%len(segList_meg))
        segList_mergeMMP = mednum_merge(segList_meg, MES_merge,y, x)
        print("合并后分段长度：%d"%len(segList_mergeMMP))
        #segList_mergeMMP = segList_meg
        #合并后的分段点复原
        ses_merge = []
        for i in range(len(segList_mergeMMP)):
            if (segList_mergeMMP[i] != None):
                ses_merge.append(segList_mergeMMP[i])
        #将ses_merge分段点复原成原时间序列顺序
        ses_merge_1 = []
        for i in range(len(y)):
            if(y[i] in ses_merge):
                ses_merge_1.append(y[i])
        #获取x轴时间戳
        time_temp=[]
        for i in range(len(ses_merge_1)):
            if ses_merge_1[i] not in time_temp:
                time_temp.append(ts_x[y.tolist().index(ses_merge_1[i])])
        
        x1 = []
        y1 = []
        for i in range(len(ses_merge_1)):
            y1.append(ses_merge_1[i])
            x1.append(x[y.tolist().index(y1[i])])
        #复原分段点
        for i in range(len(ses_merge_1)):
            ses_merge_1[i] = ses_merge_1[i] * std_TS_1 + mean_TS_1
        segList_mergeMMP = ses_merge_1
        ans_seg_point_date = time_temp
    # =============================================================================
    #     plt.figure(figsize=(12,6))
    #     plt.plot(ts_x,ts,'r--',label='raw_data') 
    #     plt.scatter(ans_seg_point_date,segList_mergeMMP,color='blue',label='seg_point')
    #     plt.plot(ans_seg_point_date,segList_mergeMMP,color='green',label='fitting_line')
    #     plt.legend()
    #     plt.xticks(ts_x, ts_x, rotation=90, fontsize=10)
    #     plt.show() 
    # =============================================================================

        error = md.calculate_fitting_error(x1,y1,y,x)
        error2 = md.calculate_vertical_error(x1,y1,y,x)
        
        
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(111)
        ax.plot(ts_x,ts,'r--',label='raw_data')
        ax.scatter(ans_seg_point_date,segList_mergeMMP,color='blue',label='seg_point')
        ax.plot(ans_seg_point_date,segList_mergeMMP,color='green',label='fitting_line')
        tick_spacing = 120
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing ))
        ax.legend()
        plt.show()
        stop_time = time.clock()
        print("the MS-BU mean square fitting error1: ",error)
        print("the MS-BU vertical fitting error2: ",error2)
        print('Running time: %s Seconds'%(stop_time-start_time))
         #将得到结果存入结果列表中
        list_error.append(error2)
        list_segNum.append(len(segList_mergeMMP))
        list_time.append(stop_time-start_time)
        MES += 0.1
    list_ans.append(list_error)
    list_ans.append(list_segNum)
    list_ans.append(list_time)    
    np.savetxt('E:/论文/论文初写/code/output/CSIW/CSIW_MS_BU.txt',list_ans,fmt='%.4f')                    
