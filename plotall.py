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
        'size'   : 12}

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

from averagemandi import mandiarrivalseries
from averagemandi import specificarrivalseries
from averagemandi import expectedarrivalseries
mean_arr_series = mandiarrivalseries
mean_arr_series = mean_arr_series.rolling(window=30).mean()
mean_arr_series = RemoveNaNFront(mean_arr_series)
#mean_arr_series = (mean_arr_series - mean_arr_series.mean())/mean_arr_series.std()

specificarrivalseries = specificarrivalseries.rolling(window=30).mean()
specificarrivalseries = RemoveNaNFront(specificarrivalseries)

from averagemandi import mandipriceseries
from averagemandi import specificpriceseries
mean_mprice_series = mandipriceseries


from averageretail import retailpriceseries
from averageretail import specificretailprice
specificretailprice = specificretailprice.rolling(window=7).mean()
mean_retail_series = retailpriceseries
mean_retail_series = mean_retail_series.rolling(window=7).mean()
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
rainfallseries = rainfallmonthly
#rainfallseries = (rainfallseries - rainfallseries.mean())/rainfallseries.std() 

from fuelprice import fuelpricedelhi
from fuelprice import fuelpricemumbai


#difference_series = (mean_retail_series - mean_mprice_series)
#ifference_series = difference_series.rolling(window = 7).mean()
#plot_series(difference_series,'Difference Retail Mandi Mean Price',6)

#ratio_series = (mean_retail_series/mean_mprice_series)
#ratio_series = ratio_series.rolling(window = 7).mean()
#plot_series(ratio_series,'Ratio Retail Mandi Mean Price',6)


#plt.title('Ratio Series')


fig, ax1 = plt.subplots()
ax1.set_xlabel('Time')
ax1.set_ylabel('Mean Price per Quintal')
plot_series_axis(mean_retail_series,'Mean Retail Price',3,ax1)
plot_series_axis(mean_mprice_series,'Mean Mandi Price',5,ax1)
ax2 = ax1.twinx()
ax2.set_ylabel('Mean Arrival in Tonnes')
plot_series_axis(mean_arr_series,'Mean Arrival Series',4,ax2)
plot_series_axis(expectedarrivalseries,'General Arrival Trend',6,ax2)
fig.tight_layout()
ax1.legend(loc = (0.05,0.9), frameon = False)
ax2.legend(loc = (0.05,0.75), frameon = False)


#plt.ylabel('Mumbai Pimpalgaon Series')
#plot_series(specificretailprice,'Mean Retail Price',3)
#plot_series(specificpriceseries,'Mean Mandi Price',5)
#plot_series(specificarrivalseries,'Mean Arrival Series',4)


#plt.ylabel('Individual Price Series')
#plot_series(specificpriceseries,'Mandi Price Pimpalgaon',6)
#plot_series(rainfallseries,'Rainfall',5)

#plt.show()
# plot_year_series_avg(rainfallseries,'Average Yearly',1)
# plot_series_year(rainfallseries,'Rainfall',2)

#plot_series(mean_rain_series,'Rainfall',4)
#plot_series(specificarrivalseries,'Pimpalgaon Arrival',5)
#plot_series(fuelpricemumbai,'Fuel Price Mumbai',6)
#plot_series(specificretailprice,'Retail Price Mumbai',6)
#plot_series(exportseries,'Export',6)
# plt.legend(loc='best')
plt.show()
