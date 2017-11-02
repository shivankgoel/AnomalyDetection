from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib
import matplotlib.pyplot as plt
import math

from rainfallmonthly import rainfallmonthly
from averagemandi import specificarrivalseries
from averagemandi import specificpriceseries
from averageretail import specificretailprice


matplotlib.rcParams.update({'font.size': 22})


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

mean_arr_series = specificarrivalseries
# mean_arr_series = mean_arr_series.rolling(window=30).mean()
# mean_arr_series = RemoveNaNFront(mean_arr_series)

def delay_corr(datax,datay,min,max):
	delay_corr_values=[]
	delays = []
	for delay in range(min,max+1):
		curr_corr = datax.corr(datay.shift(delay*30))
		delay_corr_values.append(curr_corr)
		delays.append(delay)
	plt.plot(delays,delay_corr_values)
	plt.show()
# ------------------------------------------------------------------------------------

def delay_corr2(datax,datay,min,max):
	delay_corr_values_early_kharif=[]
	delay_corr_values_kharif=[]
	delay_corr_values_rabi =[]
	delays = []
	for delay in range(min,max+1):
		new_datay = datay.shift(delay)
		curr_corr_early = 0
		for year in range(1,8):
			short_y = new_datay[year*365+250:365*(year+1)+350]
			short_x = datax[year*365+250:365*(year+1)+350]
			curr_corr_early = curr_corr_early + short_x.corr(short_y)
		# curr_corr = datax.corr(datay.shift(delay*30))
		curr_corr_early = curr_corr_early/7
		delay_corr_values_early_kharif.append(curr_corr_early)

		curr_corr_kharif = 0
		for year in range(0,8):
			short_y = new_datay[year*365+330:365*(year+1)+100]
			short_x = datax[year*365+330:365*(year+1)+100]
			curr_corr_kharif = curr_corr_kharif + short_x.corr(short_y)
		# curr_corr = datax.corr(datay.shift(delay*30))
		curr_corr_kharif = curr_corr_kharif/8
		delay_corr_values_kharif.append(curr_corr_kharif)

		curr_corr_rabi = 0
		for year in range(1,9):
			short_y = new_datay[year*365+100:365*(year)+240]
			short_x = datax[year*365+100:365*(year)+240]
			curr_corr_rabi = curr_corr_rabi + short_x.corr(short_y)
		# curr_corr = datax.corr(datay.shift(delay*30))
		curr_corr_rabi = curr_corr_rabi/8
		delay_corr_values_rabi.append(curr_corr_rabi)
		delays.append(delay)
	plt.plot(delays,delay_corr_values_early_kharif,color='r', label='Early Kharif')
	plt.plot(delays,delay_corr_values_kharif, color='g', label = 'Kharif')
	plt.plot(delays,delay_corr_values_rabi, color='b', label='Rabi')
	plt.xlabel('Time Shifted (Days)')
	plt.ylabel('Shifted Correlations')
	plt.legend(loc='best')
	plt.title('Shifted Correlations - Retail Prices vs Mandi Prices')
	plt.show()


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')



# delay_corr2(mean_arr_series,rainfallmonthly,-6,6)
# delay_corr2(specificpriceseries,mean_arr_series,-120,120)
delay_corr2(specificretailprice,specificpriceseries,-90,90)
# delay_corr2(specificretailprice,rainfallmonthly,-6,6)
# delay_corr2(specificpriceseries,rainfallmonthly,-6,6)
# delay_corr2(specificretailprice,mean_arr_series,-120,120)