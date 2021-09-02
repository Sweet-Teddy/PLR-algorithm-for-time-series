# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:05:01 2020

@author: Administrator
"""   '''timeStamp = 1557502800
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    print(otherStyleTime)'''

import time
import pandas as pd
import numpy as np
TS_x = [879696000,879782400,879868800,879955200,880041600,880300800,880387200, 880473600]
t1 = []
otherStyleTime = []
for i in range(len(TS_x)):   
    t1.append(time.localtime(TS_x[i]))
    otherStyleTime.append(time.strftime('%Y/%m/%d',t1[i]))
out = np.array(otherStyleTime)
    