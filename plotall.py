from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib
import matplotlib.pyplot as plt
import math

font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000', '#900c3f' ]

def plot_year_series_avg(inpseries,lbl,cl):
  s = '2010-01-01'
  e = '2010-12-31'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  tempseries = inpseries[s:e]
  inpseries = inpseries.groupby([inpseries.index.month, inpseries.index.day]).mean()
  yaxis = list(inpseries[0:len(tempseries)])
  xaxis = list(tempseries.index)
  plt.plot(xaxis,yaxis, color = colors[cl] ,label=lbl)


def plot_series_year(inpseries,lbl,clr):
  s = '2010-01-01'
  e = '2010-12-31'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  inpseries = inpseries[s:e]
  yaxis = list(inpseries)
  xaxis = list(inpseries.index)
  plt.plot(xaxis,yaxis, color =colors[clr] , label=lbl)



def plot_series(inpseries,lbl,clr):
  s = '2006-01-01'
  e = '2015-01-01'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  inpseries = inpseries[s:e]
  yaxis = list(inpseries)
  xaxis = list(inpseries.index)
  plt.plot(xaxis,yaxis, color =colors[clr] , label=lbl)


def plot_series_axis(inpseries,lbl,clr,ax):
  s = '2006-01-01'
  e = '2015-01-01'
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  inpseries = inpseries[s:e]
  yaxis = list(inpseries)
  xaxis = list(inpseries.index)
  ax.plot(xaxis,yaxis, color =colors[clr] , label=lbl)

  

def RemoveNaNFront(series):
  index = 0
  while True:
    if(not np.isfinite(series[index])):
      index += 1
    else:
      break
  if(index < len(series)):
    for i in range(0, index):
      series[i] = series[index]
  return series

'''
def Normalize(series):
  series = (series - series.mean())/series.std()
'''

from averagemandi import mandiarrivalseries
from averagemandi import specificarrivalseries
from averagemandi import expectedarrivalseries
from averagemandi import expectedspecificarrivalseries
mean_arr_series = mandiarrivalseries
mean_arr_series = mean_arr_series.rolling(window=30,center=True).mean()
mean_arr_series = RemoveNaNFront(mean_arr_series)
#mean_arr_series = (mean_arr_series - mean_arr_series.mean())/mean_arr_series.std()

specificarrivalseries = specificarrivalseries.rolling(window=30,center=True).mean()
specificarrivalseries = RemoveNaNFront(specificarrivalseries)

from averagemandi import mandipriceseries
from averagemandi import specificpriceseries
from averagemandi import expectedmandiprice
mean_mprice_series = mandipriceseries


from averageretail import retailpriceseries
from averageretail import specificretailprice
specificretailprice = specificretailprice.rolling(window=7,center=True).mean()
mean_retail_series = retailpriceseries
mean_retail_series = mean_retail_series.rolling(window=7,center=True).mean()
#mean_retail_series = (mean_retail_series - mean_retail_series.mean())/mean_retail_series.std()


from averagerainfall import meanrainfallseries
mean_rain_series = meanrainfallseries
mean_rain_series = mean_rain_series['2007-01-01':'2012-12-24'] - (mean_rain_series['2007-01-01':'2012-12-24']).mean()
mean_rain_series = mean_rain_series * 100
mean_rain_series = mean_rain_series['2007-01-01':'2012-12-24'] + 1000


from average_export import exportseries
#exportseries = exportseries/100
exportseries = (exportseries - exportseries.mean())/exportseries.std()

from rainfallmonthly import rainfallmonthly
from rainfallmonthly import avgrainfallmonthly
from rainfallmonthly import avgrainfallexpected
rainfallseries = rainfallmonthly
#rainfallseries = (rainfallseries - rainfallseries.mean())/rainfallseries.std() 

from fuelprice import fuelpricedelhi
from fuelprice import fuelpricemumbai


#difference_series = (mean_retail_series - mean_mprice_series)
#ifference_series = difference_series.rolling(window = 7,center=True).mean()
#plot_series(difference_series,'Difference Retail Mandi Mean Price',6)

#ratio_series = (mean_retail_series/mean_mprice_series)
#ratio_series = ratio_series.rolling(window = 7,center=True).mean()
#plot_series(ratio_series,'Ratio Retail Mandi Mean Price',6)


#plt.title('Arrival')

import numpy as np


#gapseries = retailpriceseries - mandipriceseries
# gapseries = (retailpriceseries*1.0/mandipriceseries)
# coefficients, residuals, _, _, _ = np.polyfit(range(len(gapseries.index)),gapseries,1,full=True)
# mse = residuals[0]/(len(gapseries.index))
# nrmse = np.sqrt(mse)/(gapseries.max() - gapseries.min())
# trendline = pd.Series(([coefficients[0]*x + coefficients[1] for x in range(len(gapseries))]),index=gapseries.index)
# print('Slope ' + str(coefficients[0]))
# print('NRMSE: ' + str(nrmse))

#plot_series(gapseries,'Retail - Mandi Price',5)
#plot_series(ratio_series,'Retail/Mandi Price',6)

# fig, ax1 = plt.subplots()
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Arrival')
# ax2 = ax1.twinx()
# ax2.set_ylabel('Price')
#plot_series_axis(gapseries,'Retail / Mandi Price',5,ax1)
#plot_series_axis(trendline,'Trendline',3,ax1)
#plot_series_axis(ratio_series,'Retail / Mandi Price',3,ax2)
#plot_series_axis(mandipriceseries ,'Mandi Price',6,ax2)
#plot_series_axis(retailpriceseries ,'Retail Price',5,ax2)
#plot_series_axis(mean_arr_series ,'Arrival',4,ax1)
#plot_series_axis(expectedarrivalseries,'Expected Arrival ',2,ax1)
#plot_series_axis(expectedspecificarrivalseries,'Expected Delhi Arrival ',7,ax1)
#plot_series_axis(specificarrivalseries,'Delhi Arrival ',8,ax1)

#ma = expectedmandiprice.rolling(30,center=True).mean()
#mstd = expectedmandiprice.rolling(30,center=True).std()
#ax2.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color=colors[4], alpha=0.1)


#ma = expectedarrivalseries.rolling(30,center=True).mean()
#mstd = expectedarrivalseries.rolling(30,center=True).std()
#ax1.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color=colors[2], alpha=0.1)

# fig.tight_layout()
#ax1.legend(loc = (0.05,0.9), frameon = False)
#ax2.legend(loc = (0.05,0.80), frameon = False)
'''

# mean_retail_series.to_csv('mean_retail_price.csv', header=None, encoding='utf-8')
# mean_mprice_series.to_csv('mean_mandi_price.csv', header=None, encoding='utf-8')
# mean_arr_series.to_csv('mean_arrival.csv', header=None, encoding='utf-8')
# expectedarrivalseries.to_csv('expected_arrival.csv', header=None, encoding='utf-8')

#plot_series(mean_arr_series,'Mean Arrival Series',4)
#expectedarrivalseries = (expectedarrivalseries - expectedarrivalseries.mean())/expectedarrivalseries.std()
#plot_series(expectedarrivalseries,'Expected Arrival Trend',6)

#plt.xlabel('Time')
#plt.ylabel('Arrival in Tonnes')
#plot_series(specificretailprice,'Mean Retail Price',3)
#plot_series(specificpriceseries,'Mean Mandi Price',5)
#from averagemandi import expectedspecificarrivalseries
#specificarrivalseries = (expectedspecificarrivalseries - expectedspecificarrivalseries.mean())/expectedspecificarrivalseries.std()
#plot_series(specificarrivalseries,'Mean Arrival Series',4)


#plt.ylabel('Individual Price Series')
#plot_series(specificpriceseries,'Mandi Price Pimpalgaon',6)
#plot_series(rainfallseries,'Rainfall',5)

#plt.show()
#plot_year_series_avg(rainfallseries,'Average Yearly',1)
#plot_series_year(rainfallseries,'Rainfall',2)

#plot_series(rainfallseries,'Rainfall',2)
#avgrainfallexpected = (avgrainfallexpected - avgrainfallexpected.mean())/avgrainfallexpected.std()
#plot_series(avgrainfallexpected,'Average Yearly Expected',1)
#ma = avgrainfallexpected.rolling(14,center=True).mean()
#mstd = avgrainfallexpected.rolling(14,center=True).std()

#plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color='b', alpha=0.2)

#plot_series(mean_rain_series,'Rainfall',4)
gapseries = retailpriceseries - mandipriceseries
ratio_series = (retailpriceseries*1.0/mandipriceseries)
plot_series(gapseries,'Retail - Mandi Price',5)
plot_series(ratio_series,'Retail / Mandi Price',6)
'''
#plot_series(fuelpricemumbai,'Fuel Price Mumbai',6)
#plot_series(specificretailprice,'Retail Price Mumbai',6)
#plot_series(exportseries,'Export',6)
plot_series(mandipriceseries ,'Mandi Price',6)
plot_series(retailpriceseries ,'Retail Price',5)
plt.legend(loc='best')
plt.show()
