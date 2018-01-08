from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates


font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 21}

matplotlib.rc('font', **font)

colors = [ '#e6194b', '#3cb44b' ,'#ffe119', '#0082c8', '#f58231', '#911eb4',  '#000080' , '#800000'  ,'#000000', '#900c3f' ]


def plot_series(inpseries,lbl,clr):
  s = '2006-01-01'
  e = '2015-01-01'
  s = CONSTANTS['STARTDATE']
  e = CONSTANTS['ENDDATE']
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  inpseries = inpseries[s:e]
  yaxis = list(inpseries)
  xaxis = list(inpseries.index)
  plt.plot(xaxis,yaxis, color =colors[clr] , label=lbl, linewidth=2.0)


def plot_series_axis(inpseries,lbl,clr,ax):
  s = '2006-01-01'
  e = '2015-01-01'
  s = CONSTANTS['STARTDATE']
  e = CONSTANTS['ENDDATE']
  s = datetime.strptime(s,'%Y-%m-%d')
  e = datetime.strptime(e,'%Y-%m-%d')
  inpseries = inpseries[s:e]
  yaxis = list(inpseries)
  xaxis = list(inpseries.index)
  ax.plot(xaxis,yaxis, color =colors[clr] , label=lbl, linewidth=2.0)

  

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


def Normalize(series):
  series = (series - series.mean())/series.std()
  return series

from averagemandi import mandiarrivalseries
from averagemandi import expectedarrivalseries
from averagemandi import mandipriceseries
from averagemandi import expectedmandiprice

from averagemandi import expectedspecificarrivalseries
from averagemandi import specificarrivalseries
from averagemandi import specificpriceseries

from averageretail import retailpriceseries
from averageretail import specificretailprice
from averageretail import expectedretailprice

from average_export import exportseries
from average_export import expectedexportseries

#Rainfall of Madhya Maharastra
from rainfallmonthly import rainfallmonthly
from rainfallmonthly import rainfallexpected
#Avg Rainfall of 3 regions
from rainfallmonthly import avgrainfallmonthly
from rainfallmonthly import avgrainfallexpected
# from averagerainfall import meanrainfallseries


from fuelprice import fuelpricedelhi
from fuelprice import fuelpricemumbai

from oilmonthlyseries import oilmonthlyseries
from cpimonthlyseries import cpimonthlyseries
from cpimonthlyseries import avgcpiseries

'''
0 Red
1 Green
2 Yellow
3 Light Blue
4 Orange
5 Purple
6 Dark Blue
7 Maroon
'''

def give_average_series(start,end,mandiarrivalseries):
  mandiarrivalexpected = mandiarrivalseries.rolling(window=30,center=True).mean()
  mandiarrivalexpected = mandiarrivalexpected.groupby([mandiarrivalseries.index.month, mandiarrivalseries.index.day]).mean()
  idx = pd.date_range(start, end)
  data = [ (mandiarrivalexpected[index.month][index.day]) for index in idx]
  expectedarrivalseries = pd.Series(data, index=idx)
  return expectedarrivalseries


# import matplotlib.dates as mdates
# ax1.xaxis.set_major_locator(mdates.MonthLocator(1))
# ax2.xaxis.set_major_locator(mdates.MonthLocator(1))
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,2,3,4,5,6,7,8,9,10,11,12),bymonthday=1))
# ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,2,3,4,5,6,7,8,9,10,11,12),bymonthday=1))
# ax1.legend(loc = (0.05,0.9), frameon = False)
# ax2.legend(loc = (0.05,0.85), frameon = False)

# plt.show()

# START = '2006-01-01'
# END = '2015-06-23'

# from averagemandi import getmandi
# pricemum = give_average_series(START,END,getmandi('Mumbai',False))
# pricel = give_average_series(START,END,getmandi('Lasalgaon',False))
# pricedel = give_average_series(START,END,getmandi('Azadpur',False))
# pricepune = give_average_series(START,END,getmandi('Pune',False))

# pricemum = give_average_series(start,end,pricemum)
# pricel = give_average_series(start,end,pricel)
# pricedel = give_average_series(start,end,pricedel)
# pricepune = give_average_series(start,end,pricepune)


# fig, ax1 = plt.subplots()
# #ax1.set_title('Mandi Price at Different Mandis')
# ax1.set_xlabel('Months')
# ax1.set_ylabel('Arrival in MT')
# #plot_series_axis(pricemum,'Mumbai',1,ax1)
# plot_series_axis(pricel,'Lasalgaon',4,ax1)
# plot_series_axis(pricedel,'Azadpur',6,ax1)
# #plot_series_axis(pricepune,'Pune',3,ax1)
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
# ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,2,3,4,5,6,7,8,9,10,11,12),bymonthday=1))
# ax1.legend(loc='best')
# plt.show()
#plt.title('Mandi vs Retail Price')
# plt.title('Mandi Price')
# plt.xlabel('Time')
# plt.ylabel('Price per Quintal')
#plot_series(retailpriceseries,'Retail Price',3)
#plot_series(expectedmandiprice,'Average Mandi Price',3)
#plot_series(mandipriceseries,'Mandi Price',6)
#plot_series(oilmonthlyseries,'Fuel',6)
# plot_series(rainfallmonthly,'Rainfall',2)
# plot_series(rainfallexpected,'Average Rainfall',1)
# plt.legend(loc='best')
# plt.show()


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

# gapseries = retailpriceseries - mandipriceseries
# ratio_series = (retailpriceseries*1.0/mandipriceseries)
# plot_series(gapseries,'Retail - Mandi Price',5)
# plot_series(ratio_series,'Retail / Mandi Price',6)

#plot_series(fuelpricemumbai,'Fuel Price Mumbai',6)
#plot_series(specificretailprice,'Retail Price Mumbai',6)
#plot_series(exportseries,'Export',6)
#plot_series(mandipriceseries ,'Mandi Price',6)
#plot_series(retailpriceseries ,'Retail Price',5)
#plt.legend(loc='best')



def plotweather(start,end,averagetoo,roll):
  a = rainfallmonthly[start:end]
  b = rainfallexpected[start:end]
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Rainfall')
  plt.xlabel('Time')
  plt.ylabel('Rainfall in mm')
  plot_series(a,'Rainfall',6)
  if(averagetoo):
    plot_series(b,'Average Rainfall',4)
    ma = b.rolling(30,center=True).mean()
    mstd = b.rolling(30,center=True).std()
    plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotcpi(start,end,averagetoo,roll):
  a = cpimonthlyseries[start:end]
  b = avgcpiseries[start:end]
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Cumulative Price Index')
  plt.xlabel('Time')
  plt.ylabel('CPI')
  plot_series(a,'CPI',6)
  if(averagetoo):
    plot_series(b,'Average CPI',4)
    ma = b.rolling(30,center=True).mean()
    mstd = b.rolling(30,center=True).std()
    plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotarrival(start,end,averagetoo,roll):
  a = mandiarrivalseries[start:end]
  b = expectedarrivalseries[start:end]
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Arrival')
  plt.xlabel('Time')
  plt.ylabel('Arrival in Metric Tons(Tonnes)')
  plot_series(a,'Arrival',1)
  if(averagetoo):
    plot_series(b,'Average Arrival',2)
    ma = b.rolling(30,center=True).mean()
    mstd = b.rolling(30,center=True).std()
    plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color=colors[2], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotmandiprice(start,end,averagetoo,roll):
  a = mandipriceseries[start:end]
  b = expectedmandiprice[start:end]
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Mandi Price')
  plt.xlabel('Time')
  plt.ylabel('Mandi Price per Quintal')
  plot_series(a,'Mandi Price',6)
  if(averagetoo):
    plot_series(b,'Average Mandi Price',4)
    ma = b.rolling(30,center=True).mean()
    mstd = b.rolling(30,center=True).std()
    plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotretailprice(start,end,averagetoo,roll):
  a = retailpriceseries[start:end]
  b = expectedretailprice[start:end]
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Retail Price')
  plt.xlabel('Time')
  plt.ylabel('Retail Price per Quintal')
  plot_series(a,'Retail Price',6)
  if(averagetoo):
    plot_series(b,'Average Retail Price',4)
    ma = b.rolling(30,center=True).mean()
    mstd = b.rolling(30,center=True).std()
    plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()

def plotretailvsmandi(start,end,averagetoo,roll):
  a = retailpriceseries[start:end]
  b = mandipriceseries[start:end]
  if roll:
    a = a.rolling(window=14,center=True).mean()
    b = b.rolling(window=14,center=True).mean()
  plt.title('Retail vs Mandi price')
  plt.xlabel('Time')
  plt.ylabel('Price per Quintal')
  plot_series(a,'Retail Price',6)
  if(averagetoo):
    plot_series(b,'Mandi Price',3)
  plt.legend(loc='best')
  plt.show()

def plotsingleseries(series,title,xlabel,ylabel,start,end,averagetoo,roll):
  a = series[start:end]
  if(averagetoo):
    b = give_average_series(start,end,series)
  if roll:
    a = a.rolling(window=14,center=True).mean()
    if(averagetoo):
      b = b.rolling(window=14,center=True).mean()
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plot_series(a,title,6)
  if(averagetoo):
    plot_series(b,'Average '+title,4)
    ma = b.rolling(30,center=True).mean()
    mstd = b.rolling(30,center=True).std()
    plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color=colors[4], alpha=0.2)
  plt.legend(loc='best')
  plt.show()


def plotdoubleseries(s1,s2,x1,x2,y1,y2,start,end,avg1=False,avg2=False,roll1=False,roll2=False):
  fig, ax1 = plt.subplots()
  ax1.set_xlabel(x1)
  ax1.set_ylabel(y1)
  ax2 = ax1.twinx()
  ax2.set_xlabel(x2)
  ax2.set_ylabel(y2)
  if roll1:
    s1 = s1.rolling(window=14,center=True).mean()
  if roll2: 
    s2 = s2.rolling(window=14,center=True).mean()
  plot_series_axis(s1,y1,6,ax1)
  plot_series_axis(s2,y2,3,ax2)
  if avg2:
    a = give_average_series(s2)
    plot_series(a,'Average '+y1,4)
    ma1 = a.rolling(30,center=True).mean()
    mstd1 = a.rolling(30,center=True).std()
    ax1.fill_between(mstd1.index, ma1-2*mstd1, ma1+2*mstd1, color=colors[6], alpha=0.1)
  if avg2:
    b = give_average_series(s2)
    plot_series(b,'Average '+y2,4)
    ma2 = b.rolling(30,center=True).mean()
    mstd2 = b.rolling(30,center=True).std()
    ax2.fill_between(mstd2.index, ma2-2*mstd2, ma2+2*mstd2, color=colors[3], alpha=0.1)
  fig.tight_layout()
  ax1.legend(loc = (0.05,0.9), frameon = False)
  ax2.legend(loc = (0.05,0.80), frameon = False)
  plt.show()


def linear_reg(x,y):
  from sklearn import linear_model
  from sklearn.metrics import mean_squared_error, r2_score
  from sklearn.utils import shuffle
  x,y = shuffle(x,y)
  train_size = (int)(0.80 * len(x))
  train = x[:train_size]
  train_labels = y[:train_size]
  test = x[train_size:]
  test_labels = y[train_size:]
  regr = linear_model.LinearRegression()
  regr.fit(train.values.reshape(-1,1), train_labels)
  predicted_labels = regr.predict(test.values.reshape(-1,1))
  print('Variance score: %.2f' % r2_score(test_labels, predicted_labels ))
  print("Mean squared error: %.2f" % mean_squared_error(test_labels, predicted_labels))
  plt.plot(test, test_labels, color='blue', linewidth=3)
  plt.plot(test, predicted_labels, color='red', linewidth=3)
  plt.show()




'''
weather only: '01-06-2007','31-12-2007'

'''


pstart = '2006-03-01'
pend = '2007-03-01'
fstart = CONSTANTS['STARTDATE']
fend = CONSTANTS['ENDDATE']

# plotweather(pstart,pend,True,False)
# plotarrival(pstart,pend,True,True)
# plotmandiprice(pstart,pend,True,True)
# plotretailprice(pstart,pend,True,True)
# plotretailvsmandi(pstart,pend,True,True)
# plotsingleseries(retailpriceseries-mandipriceseries,'Difference','Time','Price per Quintal',pstart,pend,False,True )
# plotsingleseries(exportseries,'Export','Time','Export in Metric Tons',pstart,pend,False,True)
# plotcpi(fstart,fend,False,False)
# plotdoubleseries(mandipriceseries,cpimonthlyseries,'Time','Time','Mandi Price','CPI',fstart,fend)
linear_reg(cpimonthlyseries,mandipriceseries)