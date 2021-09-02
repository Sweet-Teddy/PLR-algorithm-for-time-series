# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 21:49:28 2020

@author: Administrator
"""

import matplotlib.pyplot as plt
#------------MESP=0.1---------------------
#SW上数据集上误差统计SAP500 CSI300 SAP500W CSI300W SAP500P CSI300P SAP500PW CSI300PW
mean_squar_SW_error_SAP=[36.34863901570577,26.30316935097664,25.198338075172753,26.001563256870917,26.277211866271042]
mean_segNum_SW=[35,33,22,25,19]
vert_SW_error_SAP=[141.96769138492797,119.985864597133,119.68513304623174,123.32266337610348,124.4322618578855]
run_time_SW_SAP = [0.3280868930005454,0.3894150769992848,0.32345915699988836,0.3133668589999843,0.3159359280002718 ]

#FSW上数据集上误差统计SAP500 CSI300 SAP500W CSI300W SAP500P CSI300P SAP500PW CSI300PW
mean_squar_FSW_error_SAP=[5.664826511607049,20.981801870234555,45.75306483224075,69.77928990230512,109.16440397855462]
mean_segNum_FSW=[107,41,17,11,9]
vert_FSW_error_SAP=[61.86618766728604,125.66007707102747,176.50467359157014,217.15625023028966,284.02873934479805]
run_time_FSW_SAP = [0.3321457760002886,0.33218643399959547,0.31347418500081403,0.3023669939998399,0.3024048910001511]

#SFSW上数据集上误差统计SAP500 CSI300 SAP500W CSI300W SAP500P CSI300P SAP500PW CSI300PW
mean_squar_SFSW_error_SAP=[8.084218909059356,36.440902922529354,69.82292048064278,59.28525360468993,68.59444649605238]
mean_segNum_SFSW=[92,34,11,9,6]
vert_SFSW_error_SAP=[70.48654954875659,152.02232953529912,220.68313328935403,188.60901665044267,208.2880379286905]
run_time_SFSW_SAP = [0.4594835720017727,0.4309480860001713,0.32746632999987924,0.3088149320001321,0.3059734810003647]


#OPLR-TP上数据集上误差统计SAP500 CSI300 SAP500W CSI300W SAP500P CSI300P SAP500PW CSI300PW
mean_squar_OPLR_error_SAP=[7.1870296056508645,12.298004595019547,21.682471054232778,23.707190515894858,26.86827965525825]
mean_segNum_OPLR=[168,111,78,59,45]
vert_OPLR_error_SAP=[53.60732163581305,74.25324473176755,104.67663488311169,105.02028784708105,121.71358313248173]
run_time_OPLR_SAP = [0.5789262380021682,0.4787476900019101,0.43775340899810544,0.39307827600032397,0.3817002370001319]


#MS-BU上数据集上误差统计SAP500 CSI300 SAP500W CSI300W SAP500P CSI300P SAP500PW CSI300PW
mean_squar_MS_BU_error_SAP=[4.989488724464945,10.29326064632913,8.994780483424076,10.031394091208163,12.975971613985736]
mean_segNum_MS_BU=[127,81,65,57,48]
vert_MS_BU_error_SAP=[48.30493433497183,73.01964089847722,70.19489088216922,74.85103926220802,84.49332474088946]
run_time_MS_BU_SAP = [0.7052633290004451, 0.49937371800115216, 1.4994629349999968,0.4180828849998761,0.39354764100016837]


MEPP = ['10%','20%','30%','40%','50%']
plt.figure(figsize=(12,6))
# =============================================================================
# plt.scatter(MEPP,mean_squar_SW_error_SAP,color='blue',label='SW')
# plt.plot(MEPP,mean_squar_SW_error_SAP,'--d',label='SW')
# =============================================================================

plt.scatter(MEPP,mean_squar_FSW_error_SAP,color='orange',label='FSW')
plt.plot(MEPP,mean_squar_FSW_error_SAP,'--^',label='FSW' )

plt.scatter(MEPP,mean_squar_SFSW_error_SAP,color='green',label='SFSW')
plt.plot(MEPP,mean_squar_SFSW_error_SAP,'--v',label='SFSW')

plt.scatter(MEPP,mean_squar_OPLR_error_SAP,color='red',label='OPLR_TP')
plt.plot(MEPP,mean_squar_OPLR_error_SAP,'--h',label='OPLR-TP' )

plt.scatter(MEPP,mean_squar_MS_BU_error_SAP,color='pink',label='MS-BU')
plt.plot(MEPP,mean_squar_MS_BU_error_SAP,'--*',label='MS-BU' )


plt.title("Experiment on SAP500")
plt.legend()
plt.show()
