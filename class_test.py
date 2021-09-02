# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 09:10:15 2020

@author: Administrator
"""

class Person:
    def cry(self):
        print("i can cry")
    def speak(self):
        print("i can speak:%s" %(self.word))

#-------------------定义PrioL类-------------------
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
    
    
tom = Person()
tom.cry()
tom.word = "xixxi"
tom.speak()
tom.age = 18
print(tom.age)

PrioL_1 = ListNode()
PrioL_1.value = 100
list1 = []
list1.append(PrioL_1)
