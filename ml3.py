from datetime import datetime
import pandas as pd
import numpy as np
from constants import CONSTANTS
import matplotlib.pyplot as plt
import matplotlib
import sklearn.preprocessing as preprocessing

import os
cwd = os.getcwd()

def get_anomalies(path):
	anomalies = pd.read_csv(path, header=None, index_col=None)
	anomalies[0] = [ datetime.strftime(datetime.strptime(date, '%d/%m/%Y'),'%Y-%m-%d') for date in anomalies[0]]
	anomalies[1] = [ datetime.strftime(datetime.strptime(date, ' %d/%m/%Y'),'%Y-%m-%d') for date in anomalies[1]]
	return anomalies


anomalies = get_anomalies(CONSTANTS['ANOMALIES_NEWSPAPER'])
#anomaliesnew = get_anomalies('data/anomaly/anomalies_extended.csv')
anomaliesdelhi = get_anomalies('data/anomaly/delhi_anomalies_new.csv')
anomalieslucknow = get_anomalies('data/anomaly/lucknow_anomalies_new.csv')
#anomalies = pd.concat([anomalies,anomaliesnew],ignore_index=True)

from averagemandi import mandipriceseries
from averagemandi import mandiarrivalseries
from averageretail import retailpriceseries
from average_export import exportseries
from rainfallmonthly import rainfallmonthly
from fuelprice import fuelpricemumbai
from cpimonthlyseries import cpimonthlyseries
from oilmonthlyseries import oilmonthlyseries

from averageretail import getcenter
retailpriceseriesdelhi = getcenter('DELHI')
retailpriceserieslucknow = getcenter('LUCKNOW')
from averagemandi import getmandi
mandipriceseriesdelhi = getmandi('Azadpur',True)
mandiarrivalseriesdelhi = getmandi('Azadpur',False)
mandipriceserieslucknow = getmandi('Devariya',True)
mandiarrivalserieslucknow = getmandi('Devariya',False)


START = CONSTANTS['STARTDATE']
END = CONSTANTS['ENDDATEOLD']
retailpriceseries = retailpriceseries[START:END]
mandipriceseries = mandipriceseries[START:END]
mandiarrivalseries = mandiarrivalseries[START:END]
retailpriceseriesdelhi = retailpriceseriesdelhi[START:END]
retailpriceserieslucknow = retailpriceserieslucknow[START:END]
mandipriceseriesdelhi = mandipriceseriesdelhi[START:END]
mandipriceserieslucknow = mandipriceserieslucknow[START:END]
mandiarrivalseriesdelhi = mandiarrivalseriesdelhi[START:END]
mandiarrivalserieslucknow = mandiarrivalserieslucknow[START:END]


# d = {'retailp':retailpriceseries , 
# 	'mandip':mandipriceseries ,
# 	'mandiarr':mandiarrivalseries ,
# 	'export' : exportseries,
# 	'fuel': fuelpricemumbai,
# 	'cpi': cpimonthlyseries,
# 	'oil' : oilmonthlyseries
# 	}

# d1 = {'retailp':retailpriceseriesdelhi , 
# 	'mandip':mandipriceseriesdelhi ,
# 	'mandiarr':mandiarrivalseriesdelhi ,
# 	'export' : exportseries,
# 	'fuel': fuelpricemumbai,
# 	'cpi': cpimonthlyseries,
# 	'oil' : oilmonthlyseries
# 	}

# d2 = {'retailp':retailpriceserieslucknow , 
# 	'mandip':mandipriceserieslucknow,
# 	'mandiarr':mandiarrivalserieslucknow ,
# 	'export' : exportseries,
# 	'fuel': fuelpricemumbai,
# 	'cpi': cpimonthlyseries,
# 	'oil' : oilmonthlyseries
# 	}


d = {'retailp':retailpriceseries , 
	'mandip':mandipriceseries ,
	'mandiarr':mandiarrivalseries ,
	}

d1 = {'retailp':retailpriceseriesdelhi , 
	'mandip':mandipriceseriesdelhi ,
	'mandiarr':mandiarrivalseriesdelhi ,
	}

d2 = {'retailp':retailpriceserieslucknow , 
	'mandip':mandipriceserieslucknow,
	'mandiarr':mandiarrivalserieslucknow ,
	}



def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

xdf = normalize(pd.DataFrame(data=d,index=retailpriceseries.index))
xdfdelhi = normalize(pd.DataFrame(data=d1,index=retailpriceseries.index))
xdflucknow = normalize(pd.DataFrame(data=d2,index=retailpriceseries.index))


def give_anomaly_labels(xdf,anomalies,category):
	x = []
	y = []
	for anomaly_num in range(0,len(anomalies)):	
		startdate = anomalies.iloc[anomaly_num][0]
		enddate = anomalies.iloc[anomaly_num][1]
		labels = anomalies.iloc[anomaly_num][2]
		labels = labels.strip(' ').split(' ')
		if(category == ''):
			for label in labels:
				if(label == 'Hoarding'):
					y.append(1)
					x.append(preprocessing.scale(xdf[startdate:enddate]).flatten().tolist())
				if(label == 'Weather'):
					y.append(2)
					x.append(preprocessing.scale(xdf[startdate:enddate]).flatten().tolist())
				if(label == 'Inflation'):
					y.append(3)
					x.append(preprocessing.scale(xdf[startdate:enddate]).flatten().tolist())
				if(label == 'Fuel'):
					y.append(4)
					x.append(preprocessing.scale(xdf[startdate:enddate]).flatten().tolist())
				if(label == 'Transport'):
					y.append(5)
					x.append(preprocessing.scale(xdf[startdate:enddate]).flatten().tolist())
		elif(category in labels):
			y.append(1)
			x.append(preprocessing.scale(xdf[startdate:enddate]).flatten().tolist())
		else:
			y.append(0)
			x.append(preprocessing.scale(xdf[startdate:enddate]).flatten().tolist())
	return np.array(x),np.array(y)



x,y = give_anomaly_labels(xdf,anomalies,'')
x1,y1 = give_anomaly_labels(xdfdelhi,anomaliesdelhi,'')
x2,y2 = give_anomaly_labels(xdflucknow,anomalieslucknow,'')

# t = int(0.2*y.shape[0])
# t1 = int(0.2*y1.shape[0])
# t2 = int(0.2*y2.shape[0])
# xtrain = np.concatenate([np.concatenate([x[0:-t],x1[0:-t1]]),x2[0:-t2]])
# xtest = np.concatenate([np.concatenate([x[-t:],x1[-t1:]]),x2[-t2:]])
# ytrain = np.concatenate([np.concatenate([y[0:-t],y1[0:-t1]]),y2[0:-t2]])
# ytest = np.concatenate([np.concatenate([y[-t:],y1[-t1:]]),y2[-t2:]])


def givetraintest(x,y,testidx):
	xtest = np.array([x[i] for i in testidx])
	xtrain = np.array([x[i] for i in range(0,len(x)) if i not in testidx])
	ytest = np.array([y[i] for i in testidx])
	ytrain = np.array([y[i] for i in range(0,len(x)) if i not in testidx])
	return xtest,xtrain,ytest,ytrain

# xtest,xtrain,ytest,ytrain = givetraintest(x,y,[29,27,22,19,9])
# xtest1,xtrain1,ytest1,ytrain1 = givetraintest(x1,y1,[13,24,39,50,55])
# xtest2,xtrain2,ytest2,ytrain2 = givetraintest(x2,y2,[10,21,22,26])

xtest,xtrain,ytest,ytrain = givetraintest(x,y,[4,29,27,26,22,19,9,11,1])
xtest1,xtrain1,ytest1,ytrain1 = givetraintest(x1,y1,[13,24,39,50,55,47,29,33,16,5,21,39,9])
xtest2,xtrain2,ytest2,ytrain2 = givetraintest(x2,y2,[10,21,22,26,24])


def concat(a,b,c):
	return np.concatenate([np.concatenate([a,b]),c])

xtest = concat(xtest,xtest1,xtest2)
xtrain = concat(xtrain,xtrain1,xtrain2)
ytest = concat(ytest,ytest1,ytest2)
ytrain = concat(ytrain,ytrain1,ytrain2)


# from sklearn.decomposition import PCA
# pca = PCA(n_components=10, whiten=True)
# xtrain = pca.fit_transform(xtrain)
# xtest = pca.transform(xtest)


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# tuned_parameters = [{'n_estimators': [1,2,5,10,20,50,100]}]
# scores = ['precision', 'recall']
# from sklearn.model_selection import GridSearchCV
# for score in scores:    
#     #model = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring='%s_macro' % score)
#     model = GridSearchCV(RandomForestClassifier(), tuned_parameters,scoring='%s_macro' % score)
#     model.fit(xtrain,ytrain)
#     test_pred = model.predict(xtest)
#     train_pred = model.predict(xtrain)
#     from sklearn.metrics import confusion_matrix
#     cfmatrix1 = confusion_matrix(ytest,test_pred)
#     cfmatrix2 = confusion_matrix(ytrain,train_pred)
#     print cfmatrix1
#     print cfmatrix2
# print("Best parameters set found on development set:")
# print(model.best_params_)

#model = SVC(kernel= 'rbf', C= 100, gamma= 0.0001)
#model = SVC(kernel='linear', C=10)
model = RandomForestClassifier(n_estimators=10)
#model = RandomForestClassifier()
model.fit(xtrain,ytrain)
test_pred = model.predict(xtest)
train_pred = model.predict(xtrain)
from sklearn.metrics import confusion_matrix
cfmatrix1 = confusion_matrix(ytest,test_pred)
cfmatrix2 = confusion_matrix(ytrain,train_pred)
print cfmatrix1
print cfmatrix2

    
