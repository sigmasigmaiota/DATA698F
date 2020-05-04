# -*- coding: utf-8 -*-
"""
Created on Mon May  4 02:01:47 2020

@author: steph
"""

import pandas as pd
import matplotlib.pyplot as plt

###################################################################################### 100th iteration

timeseriesHIV = []
timeseriesPrEP = []

def run_analysis(num):
    
    pdataBL = pd.read_csv('C:/Users/steph/Documents/DATA698/Data/Baseline2/NE_ABM_Baseline2.csv')
    pdata1 = pd.read_csv('C:/Users/steph/Documents/DATA698/Data/Iteration1BL2_HETpen0.125/NE_ABM_100stps_5000ppl_75hw.csv')
    pdata2 = pd.read_csv('C:/Users/steph/Documents/DATA698/Data/Iteration2BL2_HETpen0.125/NE_ABM_100stps_5000ppl_75hw.csv')
    pdata3 = pd.read_csv('C:/Users/steph/Documents/DATA698/Data/Iteration3BL2_HETpen0.125/NE_ABM_100stps_5000ppl_75hw.csv')
    pdata4 = pd.read_csv('C:/Users/steph/Documents/DATA698/Data/Iteration4BL2_HETpen0.125/NE_ABM_100stps_5000ppl_75hw.csv')
    pdata5 = pd.read_csv('C:/Users/steph/Documents/DATA698/Data/Iteration5BL2_HETpen0.125/NE_ABM_100stps_5000ppl_75hw.csv')


    pdata1 = pdata1.iloc[(num*5000)-5000:(num*5000),:]
    pdata2 = pdata2.iloc[(num*5000)-5000:(num*5000),:]
    pdata3 = pdata3.iloc[(num*5000)-5000:(num*5000),:]
    pdata4 = pdata4.iloc[(num*5000)-5000:(num*5000),:]
    pdata5 = pdata5.iloc[(num*5000)-5000:(num*5000),:]
    
    pdtBL = pdataBL.groupby(['MALE','RACEETH','SAMESEXANY'])['HIV','PrEP'].mean().reset_index().add_suffix('_BL')
    pdt1 = pdata1.groupby(['Male','RaceEth','SexMinority'])['HIVstat','PrEPRx'].mean().reset_index().add_suffix('_1')
    pdt2 = pdata2.groupby(['Male','RaceEth','SexMinority'])['HIVstat','PrEPRx'].mean().reset_index().add_suffix('_2')
    pdt3 = pdata3.groupby(['Male','RaceEth','SexMinority'])['HIVstat','PrEPRx'].mean().reset_index().add_suffix('_3')
    pdt4 = pdata4.groupby(['Male','RaceEth','SexMinority'])['HIVstat','PrEPRx'].mean().reset_index().add_suffix('_4')
    pdt5 = pdata5.groupby(['Male','RaceEth','SexMinority'])['HIVstat','PrEPRx'].mean().reset_index().add_suffix('_5')
    
    frames = [pdtBL, pdt1, pdt2, pdt3, pdt4, pdt5]
    
    pdata = pd.concat(frames, axis=1)
    
    analysis1 = pdata.loc[:,['Male_1', 'RaceEth_1', 'SexMinority_1',
                             'HIVstat_1','HIVstat_2','HIVstat_3','HIVstat_4','HIVstat_5',
                             'PrEPRx_1','PrEPRx_2','PrEPRx_3','PrEPRx_4','PrEPRx_5']]
    
    analysis1['PrEP'] =  analysis1.iloc[:,8:13].sum(axis=1)/5
    
    analysis1_corr1 = 8*[analysis1.iloc[0:8,4].mean()]
    analysis1_corr2 = 8*[analysis1.iloc[8:,4].mean()]
    analysis1_corr = analysis1_corr2 + analysis1_corr1
    
    analysis1['HIV'] =  analysis1.iloc[:,3:8].sum(axis=1)/5
    
    analysis1 = analysis1.loc[:,['Male_1', 'RaceEth_1', 'SexMinority_1','HIV','PrEP']]
    
    plt.figure(figsize = [10,8])
    p1 = plt.barh(analysis1.index,analysis1['HIV'], alpha = 0.5, color = '#DE90E8')
    #plt.scatter(analysis1_corr, analysis1.index, alpha = 0.9, color="black")
    p2 = plt.barh(analysis1.index, analysis1_corr, alpha = 0.5, color="#87C1FF")
    plt.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
               ['F OTH STR','F OTH GB','F BL STR','F BL GB','F WH STR','F WH GB',
                'F LAT STR','F LAT GB','M OTH STR','M OTH GB','M BL STR','M BL GB',
                'M WH STR','M WH GB','M LAT STR','M LAT GB'])
    plt.title('HIV and PrEP, NE '+str(num)+'th Iteration')
    plt.xlabel('Proportion')
    plt.legend((p1[0],p2[0]),('HIV','PrEP'))
    plt.savefig('C:/Users/steph/Documents/DATA698/Visualizations/HIV_PrEP_Baseline2_HETpen0.125_'+str(num)+'.png', quality = 90, format = 'png', dpi=600)
    plt.show()
    
    analysis1.to_csv('C:/Users/steph/Documents/DATA698/Data/Steps100_HIV_PrEP_Baseline2_HETpen0.125_'+str(num)+'.csv',index=False)
    
    trans = analysis1.T.rename(columns={0:'F_OTH_STR',1:'F_OTH_GB',2:'F_BL_STR',3:'F_BL_GB',4:'F_WH_STR',5:'F_WH_GB', 
                                        6:'F_LAT_STR',7:'F_LAT_GB',8:'M_OTH_STR',9:'M_OTH_GB',10:'M_BL_STR',11:'M_BL_GB',
                                        12:'M_WH_STR',13:'M_WH_GB',14:'M_LAT_STR',15:'M_LAT_GB'}).reset_index(drop=True)
    
    dictHIV = {}
    dictPrEP = {}
    
    transHIV = trans.iloc[3,:].to_dict()
    transPrEP = trans.iloc[4,:].to_dict()
    
    dictHIV.update(transHIV)
    dictPrEP.update(transPrEP)
    
    timeseriesHIV.append(dictHIV)
    timeseriesPrEP.append(dictPrEP)
    
    
    return timeseriesHIV, timeseriesPrEP

############################################################ plot HIV total percentage growth

for num in range(10,101,10):
    run_analysis(num)

timeseriesHIVdf = pd.DataFrame(timeseriesHIV)
timeseriesPrEPdf = pd.DataFrame(timeseriesPrEP)
    
timeseriesHIVdf.to_csv('C:/Users/steph/Documents/DATA698/Data/Timeseries_HIV_Baseline2_HETpen0.125.csv',index=False)
timeseriesPrEPdf.to_csv('C:/Users/steph/Documents/DATA698/Data/Timeseries_PrEP_Baseline2_HETpen0.125.csv',index=False)

plt.figure(figsize = [10,12])
p1 = plt.plot(timeseriesHIVdf)

plt.xticks([0,1,2,3,4,5,6,7,8,9],['10','20','30','40','50','60','70','80','90','100'])


rn = [0,0,0,0,
      0,0,0,1,
      0,0,0,0,
      0,0,0,0]
    
for i, col in enumerate(timeseriesHIVdf.columns):
    plt.annotate(col,xy=(plt.xticks()[0][-1]+0.1+rn[i], 
                     timeseriesHIVdf[col].iloc[-1]),
    fontsize=7, fontweight='bold')
plt.xlim(0,11)

plt.title('PrEP, NE')
plt.xlabel('Model Incremental Steps')
plt.ylabel('Proportion')
plt.box(on=None)
plt.savefig('C:/Users/steph/Documents/DATA698/Visualizations/PrEP_Baseline2_HETpen0.125_Growth.png', quality = 90, format = 'png', dpi=600)
plt.show()

labs = ['F OTH STR','F OTH GB','F BL STR','F BL GB','F WH STR','F WH GB',
                'F LAT STR','F LAT GB','M OTH STR','M OTH GB','M BL STR','M BL GB',
                'M WH STR','M WH GB','M LAT STR','M LAT GB']

plt.figure(figsize = [10,12])
ax = plt.plot(timeseriesPrEPdf)

plt.xticks([0,1,2,3,4,5,6,7,8,9],['10','20','30','40','50','60','70','80','90','100'])

rn = [0,0,-1,-3,
      2, 1, 0, 1,
      0, 2, 2, 1.,
      0, 0, 1., 0]
    
for i, col in enumerate(timeseriesPrEPdf.columns):
    plt.annotate(col,xy=(plt.xticks()[0][-1]+0.1+rn[i], 
                     timeseriesPrEPdf[col].iloc[-1]),
    fontsize=7, fontweight='bold')
plt.xlim(0,12)
plt.title('HIV, NE')
plt.xlabel('Model Incremental Steps')
plt.ylabel('Proportion')
plt.box(on=None)
plt.savefig('C:/Users/steph/Documents/DATA698/Visualizations/HIV_Baseline2_HETpen0.125_Growth.png', quality = 90, format = 'png', dpi=600)
p1 = plt.show()