# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 20:48:48 2020

@author: Administrator"""
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
import numpy as np
import fit
import methods as md
def draw_plot(data,plot_title):
    plot(range(len(data)),data,alpha=0.8,color='red')
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((0,len(data)-1))

def draw_segments(segments):
    ax = gca()
    for segment in segments:
        line = Line2D((segment[0],segment[2]),(segment[1],segment[3]))
        ax.add_line(line)



max_error = 0.005



def slidingwindowsegment(sequence, create_segment, compute_error, max_error, seq_range=None):
    """
    Return a list of line segments that approximate the sequence.

    The list is computed using the sliding window technique. 

    Parameters
    ----------
    sequence : sequence to segment
    create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
    compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
    max_error: the maximum allowable line segment fitting error

    """
    if not seq_range:
        seq_range = (0,len(sequence)-1)

    start = seq_range[0]
    end = start
    result_segment = create_segment(sequence,(seq_range[0],seq_range[1]))
    while end < seq_range[1]:
        end += 1
        test_segment = create_segment(sequence,(start,end))
        error = compute_error(sequence,test_segment)
        if error <= max_error:
            result_segment = test_segment
        else:
            break

    if end == seq_range[1]:
        return [result_segment]
    else:
        return [result_segment] + slidingwindowsegment(sequence, create_segment, compute_error, max_error, (end-1,seq_range[1]))

if __name__=='__main__':
    start =time.clock()
    TS = md.loadDataSet(r'E:\论文\论文初写\data\test.csv')
    #TS_x = md.loadDataSet(r'C:\Users\Administrator\Desktop\论文\神经网络\美国标准普尔500指数历史数据(1990-2019).csv')
    TS_x = md.loadDataSet(r'E:\论文\论文初写\data\test.csv')
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
    