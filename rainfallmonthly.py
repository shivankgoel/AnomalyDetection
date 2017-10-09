from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from constants import CONSTANTS
import matplotlib.pyplot as plt
import math

START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATE']

def monthlyrainfallseries():
	rainfall = pd.read_csv('data/original/mh_rainfall.csv')
	rainfall[3] = rainfall.apply(lambda row: datetime.strptime(row[2], '%m-%Y'), axis=1)
	rainfall.index = rainfall[3]
	idx = pd.date_range(START, END)
	rainfall = rainfall.reindex(idx, fill_value=0)
	rainfallseries = rainfall['MADHYA MAHARASHTRA']
	rainfallseries = rainfallseries.replace(0.0, np.NaN, regex=True)
	rainfallseries = rainfallseries.interpolate(method='linear',limit_direction='both')
	return rainfallseries

rainfallmonthly = monthlyrainfallseries()


def giveavgrainfallmonthly():
	rainfall = pd.read_csv('data/original/mh_rainfall_all.csv')
	rainfall[3] = rainfall.apply(lambda row: datetime.strptime(row[4], '%m-%Y'), axis=1)
	rainfall.index = rainfall[3]
	idx = pd.date_range(START, END)
	rainfall = rainfall.reindex(idx, fill_value=0)
	rainfallseries1 = rainfall['MADHYA MAHARASHTRA']
	rainfallseries2 = rainfall['GUJARAT REGION']
	rainfallseries3 = rainfall['EAST RAJASTHAN']
	rainfallseries1 = rainfallseries1.replace(0.0, np.NaN, regex=True)
	rainfallseries1 = rainfallseries1.interpolate(method='linear',limit_direction='both')
	rainfallseries2 = rainfallseries2.replace(0.0, np.NaN, regex=True)
	rainfallseries2 = rainfallseries2.interpolate(method='linear',limit_direction='both')
	rainfallseries3 = rainfallseries3.replace(0.0, np.NaN, regex=True)
	rainfallseries3 = rainfallseries3.interpolate(method='linear',limit_direction='both')
	rainfallseries = (rainfallseries1+rainfallseries2 + rainfallseries3)/3
	return rainfallseries

avgrainfallmonthly = giveavgrainfallmonthly()