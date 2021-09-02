# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:46:12 2020

@author: Administrator
"""
#---------------------算法 4-2 伪代码----------------------
'''
                    程级创建ListNode实例
输入：当前时间序列SWi的存储列表CurTS；存放前k个转折点（按照从大到小的顺序）的队列curTPL

curTPL.sort(); #当前转折点按照其index属性（原始时序信息）
for i in range(len(curTPL.length)):
    if i == 0 :
        seg1 = createPLRSeg(vt1,prioL.get(i)) # 计算第1个分段的表示误差
        sumES += seg1.getES();  #计算整体表示误差
        if seg1.getES() > mes :
            mes = seg1.getES(); #不断更新最大分段误差
        end if
        Continue;
    end if
    if i == k-1 :
        LNk = createListNode(SegList, CurTS, sumES, mes); # LNk的创建过程，如算法4-3所示
        Break;
    end if
    seg = createPLRSeg(prioL.get(i), prioL.get(i+1));
    sumES += seg.getES();
    if seg.getES() > mes :
        mes = seg.getES();
    end if
end for
PAASegList = CreatePAARes(CurTS,m); #根据当前分段数k创建基于PAA的表示结果，生成 k+1 个PAA
#PAASeg实例并存入PAASeg对象数组PAASegList中
SAXSegList = CreateSAXRes(PAASegList); #根据当前PAA方法的表示结果PAASegList，生成k+1个SAXSeg实例并存入SAXSe9对象数组SAXSegList中
LNk.setparL(PAASegList);#将将PAASegList作为LNk的属性parL
LNk.setparL(SymbolList);#将SAXSegList作为LNk的属性sarL
ARList.add(LNk,k); #将创建完成的LNk知己插入ARList的第k个位置

'''
