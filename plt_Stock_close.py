# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:56:36 2020

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#——————————————————导入数据——————————————————————
f=open(r'C:\Users\Administrator\Desktop\论文\神经网络\test.csv')
df=pd.read_csv(f)     #读入股票数据
#df['Close'].plot()
plt.plot(df['Date'],df['Close'])
plt.show()
plt.scatter(df['Date'],df['Close'])
plt.show()
