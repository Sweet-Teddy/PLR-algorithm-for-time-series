# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:09:50 2020

@author: Administrator
"""
import matplotlib.pyplot as plt
#----------MEPP=10%-----MEPS=3×10%----------
mean_squar_error1_SW = 6.253040107408928
mean_squar_error1_FSW=0.0653268696487406
mean_squar_error1_SFSW=0.9513701302527645
mean_squar_error1_OPLR_TP=0.951120875268304
vert_error1_SW = 10.239236711913016
vert_error1_FSW=0.6339914774114106
vert_error1_SFSW=3.369796067587063
vert_error1_OPLR_TP=3.3767500301570497


#----------MEPP=20%-----MEPS=3×20%----------

mean_squar_error2_SW = 17.35443985932151
mean_squar_error2_FSW= 0.37507298068648526
mean_squar_error2_SFSW= 1.2706311780153494
mean_squar_error2_OPLR_TP= 0.9515939047393727
vert_error2_SW = 17.89709406684125
vert_error2_FSW= 1.8098043400897876
vert_error2_SFSW= 4.049086844545754
vert_error2_OPLR_TP= 3.2991727908508093


#----------MEPP=30%-----MEPS=3×30%----------

mean_squar_error3_SW = 27.542783227583882
mean_squar_error3_FSW= 0.6275208713760636
mean_squar_error3_SFSW= 1.8527987350067516
mean_squar_error3_OPLR_TP= 0.9129500686651811

vert_error3_SW = 23.541325435414997
vert_error3_FSW= 2.695297371241883
vert_error3_SFSW= 5.707749219354003
vert_error3_OPLR_TP= 3.390145969724119


#----------MEPP=40%-----MEPS=3×40%----------

mean_squar_error4_SW = 27.542783227583882
mean_squar_error4_FSW= 2.226527679804353
mean_squar_error4_SFSW= 2.226527679804353
mean_squar_error4_OPLR_TP= 0.9129500686651811

vert_error4_SW = 23.541325435414997
vert_error4_FSW= 5.843892259752939
vert_error4_SFSW= 5.843892259752939
vert_error4_OPLR_TP= 3.390145969724119

#----------MEPP=50%-----MEPS=3×50%----------
mean_squar_error5_SW = 27.542783227583882
mean_squar_error5_FSW= 2.226527679804353
mean_squar_error5_SFSW= 2.226527679804353
mean_squar_error5_OPLR_TP= 0.9129500686651811

vert_error5_SW = 23.541325435414997
vert_error5_FSW= 5.843892259752939
vert_error5_SFSW= 5.843892259752939
vert_error5_OPLR_TP= 3.390145969724119

NARE_1_SW = [mean_squar_error1_SW,mean_squar_error2_SW,mean_squar_error3_SW,mean_squar_error4_SW,mean_squar_error5_SW]
NARE_1_FSW = [mean_squar_error1_FSW,mean_squar_error2_FSW,mean_squar_error3_FSW,mean_squar_error4_FSW,mean_squar_error5_FSW]
NARE_1_SFSW = [mean_squar_error1_SFSW,mean_squar_error2_FSW,mean_squar_error3_FSW,mean_squar_error4_FSW,mean_squar_error5_FSW]
NARE_1_OPLR_TP = [vert_error1_OPLR_TP,vert_error2_OPLR_TP,vert_error3_OPLR_TP,vert_error4_OPLR_TP,vert_error5_OPLR_TP]
MEPP = ['10%','20%','30%','40%','50%']
# =============================================================================
# for i in range(len(NARE_1_SW)):
#     NARE_1_SW[i] = round(NARE_1_SW,3)
#     NARE_1_FSW[i] = round(NARE_1_FSW,3)
#     NARE_1_SFSW[i] = round(NARE_1_SFSW,3)
#     NARE_1_OPLR_TP[i] = 
# =============================================================================
plt.figure(figsize=(12,6))
plt.scatter(MEPP,NARE_1_SW,color='blue',label='SW')
plt.plot(MEPP,NARE_1_SW,'--d',label='SW')
plt.scatter(MEPP,NARE_1_FSW,color='orange',label='FSW')
plt.plot(MEPP,NARE_1_FSW,'--^',label='FSW' )
plt.scatter(MEPP,NARE_1_SFSW,color='green',label='SFSW')
plt.plot(MEPP,NARE_1_SFSW,'--v',label='SFSW')
plt.scatter(MEPP,NARE_1_OPLR_TP,color='red',label='OPLR_TP')
plt.plot(MEPP,NARE_1_OPLR_TP,'--h',label='OPLR-TP' )
plt.legend()
plt.show()