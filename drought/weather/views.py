from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response

import pandas as pd
import pandas as DataFrame
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot

import numpy
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm
import itertools 
import sys
import ast


class WeatherView(APIView):
	def get(self, request):
		locations = request.GET.get('location')
		months = request.GET.get('month')
		dmonth = {'January':10,'February':11,'March':12,'April':1,'May':2,'June':3,'July':4,'August':5,'September':6,'October':7,'November':8,'December':9}
		valueRain=predictRainfall(locations,dmonth[months])
		valueTemp=predictTemp(locations,dmonth[months])
		valueCC=predictCC(locations,dmonth[months])
		valueEva=predictEvapo(locations,dmonth[months])
		droughtVal=club(locations,valueRain[11],valueCC[11],valueTemp[11],valueEva[11])
		DP=predictDrought(droughtVal)
		response={
		"month":dmonth[months],
		"rainfall":valueRain,
		"temp":valueTemp,
		"cc":valueCC,
		"eva":valueEva,
		"droughtP":DP
		}
		return Response(response)

def predictDrought(droughtVal):
	if(droughtVal<=7):
		DP=(droughtVal-3)*5
	if(droughtVal>7 and droughtVal<=10):
		DP=(droughtVal-7)*6.666+20
	if(droughtVal>10 and droughtVal<=13):
		DP=(droughtVal-10)*6.666+40
	if(droughtVal>13 and droughtVal<=16):
		DP=(droughtVal-13)*6.666+60
	if(droughtVal>16 and droughtVal<=19):
		DP=(droughtVal-16)*6.666+80
	return DP

def predictRainfall(val1,val2):
	df_all = pd.read_csv('HIMACHAL PRADESH-DATASET(rainfall).csv', delimiter=',')
	df_all.head()
	location=val1
	df = df_all[df_all.Area == location]
	sarima=pd.read_csv('sarima-rainfall.csv',converters={"Order": ast.literal_eval,"SeasonalOrder": ast.literal_eval})
	sarima=sarima[sarima.Area==location]
	temporder=(sarima.Order).iloc[0]
	tempseasonal_order=(sarima.SeasonalOrder).iloc[0]
	df.head()

	months_in_year = 12
	X=df["Rainfall"].values
	validation_size=12
	start=int(len(X)*0.5)
	train_size = int(len(X) -validation_size)
	prev=X[(train_size-12+val2):]
	train, test = X[start:train_size], X[train_size:]
	history = [x for x in train]
	#diff = difference(history, 12)
	predictions = list()
	tmp_mdl = sm.tsa.statespace.SARIMAX(train,order=temporder,seasonal_order=tempseasonal_order,enforce_stationarity=True,enforce_invertibility=True)
	model_fit = tmp_mdl.fit()
	yhat = float(model_fit.forecast()[0])
    
	for i in range(0,12-val2):
		predictions.append(prev[i])

	predictions.append(yhat)
	history.append(test[0])

	bias=-6.077092

	for i in range(1,  (val2)):
		tmp_mdl = sm.tsa.statespace.SARIMAX(history,order=temporder,seasonal_order=tempseasonal_order,enforce_stationarity=True,enforce_invertibility=True)
		model_fit = tmp_mdl.fit()
		yhat =model_fit.forecast()[0]
		predictions.append(yhat)
		# observation
		obs = test[i]
		history.append(obs)
		#print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	return predictions
	    
	# report performance
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)

	print('RMSE: %.3f' % rmse)
	#pyplot.plot(test)
	#pyplot.plot(predictions, color='red')
	#pyplot.show()


def predictTemp(val1,val2):
	

	df_all = pd.read_csv('HIMACHAL PRADESH-DATASET(Av Temp).csv', delimiter=',')
	df_all.head()
	location=val1

	df = df_all[df_all.Area == location]
	sarima=pd.read_csv('sarima-temp.csv',converters={"Order": ast.literal_eval,"SeasonalOrder": ast.literal_eval})
	sarima=sarima[sarima.Area==location]
	temporder=(sarima.Order).iloc[0]
	tempseasonal_order=(sarima.SeasonalOrder).iloc[0]
	df.head()

	months_in_year = 12
	X=df["Temperature"].values
	start=int(len(X)*0.5)
	validation_size=12
	train_size = int(len(X) -validation_size)
	prev=X[(train_size-12+val2):]
	train, test = X[start:train_size], X[train_size:]
	history = [x for x in train]
	#diff = difference(history, 12)
	predictions = list()

	for i in range(0,12-val2):
		predictions.append(prev[i])

	tmp_mdl = sm.tsa.statespace.SARIMAX(train,order=temporder,seasonal_order=tempseasonal_order,enforce_stationarity=True,enforce_invertibility=True)
	model_fit = tmp_mdl.fit()
	yhat = float(model_fit.forecast()[0])


	predictions.append(yhat)
	history.append(test[0])

	bias=-6.077092

	for i in range(1,  (val2)):
		tmp_mdl = sm.tsa.statespace.SARIMAX(history,order=temporder,seasonal_order=tempseasonal_order,enforce_stationarity=True,enforce_invertibility=True)
		model_fit = tmp_mdl.fit()
		yhat =model_fit.forecast()[0]
		predictions.append(yhat)
		# observation
		obs = test[i]
		history.append(obs)
		#print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	return predictions
	    
	# report performance
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)

	print('RMSE: %.3f' % rmse)
	#pyplot.plot(test)
	#pyplot.plot(predictions, color='red')
	#pyplot.show()

def predictEvapo(val1,val2):
	

	df_all = pd.read_csv('HIMACHAL PRADESH-DATASET(Eva).csv', delimiter=',')
	df_all.head()
	location=val1
	df = df_all[df_all.Area == location]
	sarima=pd.read_csv('sarima-eva.csv',converters={"Order": ast.literal_eval,"SeasonalOrder": ast.literal_eval})
	sarima=sarima[sarima.Area==location]
	temporder=(sarima.Order).iloc[0]
	tempseasonal_order=(sarima.SeasonalOrder).iloc[0]
	df.head()

	months_in_year = 12
	X=df["Evapotranspiration"].values
	start=int(len(X)*0.5)
	validation_size=12
	
	train_size = int(len(X) -validation_size)
	prev=X[(train_size-12+val2):]
	train, test = X[start:train_size], X[train_size:]
	history = [x for x in train]
	#diff = difference(history, 12)
	predictions = list()
	for i in range(0,12-val2):
		predictions.append(prev[i])

	tmp_mdl = sm.tsa.statespace.SARIMAX(train,order=temporder,seasonal_order=tempseasonal_order,enforce_stationarity=True,enforce_invertibility=True)
	model_fit = tmp_mdl.fit()
	yhat = float(model_fit.forecast()[0])

	predictions.append(yhat)
	history.append(test[0])

	bias=-6.077092

	for i in range(1,  (val2)):
		tmp_mdl = sm.tsa.statespace.SARIMAX(history,order=temporder,seasonal_order=tempseasonal_order,enforce_stationarity=True,enforce_invertibility=True)
		model_fit = tmp_mdl.fit()
		yhat =model_fit.forecast()[0]
		predictions.append(yhat)
		# observation
		obs = test[i]
		history.append(obs)
		#print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	return predictions
	    
	# report performance
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)

	print('RMSE: %.3f' % rmse)
	#pyplot.plot(test)
	#pyplot.plot(predictions, color='red')
	#pyplot.show()

def predictCC(val1,val2):
	

	df_all = pd.read_csv('HIMACHAL PRADESH-DATASET(Cloud Cover).csv', delimiter=',')
	df_all.head()
	location=val1
	df = df_all[df_all.Area == location]
	sarima=pd.read_csv('sarima-cloud.csv',converters={"Order": ast.literal_eval,"SeasonalOrder": ast.literal_eval})
	sarima=sarima[sarima.Area==location]
	temporder=(sarima.Order).iloc[0]
	tempseasonal_order=(sarima.SeasonalOrder).iloc[0]
	df.head()

	months_in_year = 12
	X=df["CC"].values

	start=int(len(X)*0.5)
	validation_size=12
	train_size = int(len(X) -validation_size)
	prev=X[(train_size-12+val2):]
	train, test = X[start:train_size], X[train_size:]
	history = [x for x in train]
	#diff = difference(history, 12)
	predictions = list()
	for i in range(0,12-val2):
		predictions.append(prev[i])	

	tmp_mdl = sm.tsa.statespace.SARIMAX(train,order=temporder,seasonal_order=tempseasonal_order,enforce_stationarity=True,enforce_invertibility=True)
	model_fit = tmp_mdl.fit()
	yhat = float(model_fit.forecast()[0])

	predictions.append(yhat)
	history.append(test[0])

	bias=-6.077092

	for i in range(1,  (val2)):
		tmp_mdl = sm.tsa.statespace.SARIMAX(history,order=temporder,seasonal_order=tempseasonal_order,enforce_stationarity=True,enforce_invertibility=True)
		model_fit = tmp_mdl.fit()
		yhat =model_fit.forecast()[0]
		predictions.append(yhat)
		# observation
		obs = test[i]
		history.append(obs)
		#print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	return predictions
	    
	# report performance
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)

	print('RMSE: %.3f' % rmse)
	#pyplot.plot(test)
	#pyplot.plot(predictions, color='red')
	#pyplot.show()

minrain = {"Chamba":-0.771,"Kangra":-0.736,"Kinnaur":-0.826,"Kullu":-0.788,"Lahul & Spiti":-0.887,"Mandi":-0.748,"shimla":-0.766,"Sirmaur":-0.724,"Solan":-0.723}
maxrain = {"Chamba":6.683,"Kangra":6.115,"Kinnaur":5.602,"Kullu":5.189,"Lahul & Spiti":5.397,"Mandi":5237,"shimla":4.991,"Sirmaur":5.345,"Solan":4.112}
mincc = {"Chamba":-2.99,"Kangra":-2.65,"Kinnaur":-3.28,"Kullu":-3.17,"Lahul & Spiti":-3.52,"Mandi":-2.71,"shimla":-2.78,"Sirmaur":-2.46,"Solan":-2.46}
maxcc = {"Chamba":2.13,"Kangra":2.18,"Kinnaur":2.8,"Kullu":2.54,"Lahul & Spiti":2.54,"Mandi":2.44,"shimla":2.67,"Sirmaur":2.47,"Solan":2.47}
minat = {"Chamba":-1.89,"Kangra":-1.86,"Kinnaur":-2.29,"Kullu":-2.29,"Lahul & Spiti":-2.01,"Mandi":-1.97,"shimla":-2.16,"Sirmaur":-2.09,"Solan":-1.95}
maxat = {"Chamba":1.55,"Kangra":1.63,"Kinnaur":1.57,"Kullu":1.57,"Lahul & Spiti":1.46,"Mandi":1.68,"shimla":1.67,"Sirmaur":1.66,"Solan":1.71}
minpe = {"Chamba":-1.79,"Kangra":-1.74,"Kinnaur":-2.82,"Kullu":-1.82,"Lahul & Spiti":-1.84,"Mandi":-1.75,"shimla":-1.78,"Sirmaur":-1.75,"Solan":-1.72}
maxpe = {"Chamba":1.82,"Kangra":1.97,"Kinnaur":2.09,"Kullu":2.06,"Lahul & Spiti":1.57,"Mandi":2.06,"shimla":2.17,"Sirmaur":2.17,"Solan":2.11}
sdrain = {"Chamba":92.3,"Kangra":92.43,"Kinnaur":97.54,"Kullu":100.33,"Lahul & Spiti":72.05,"Mandi":96.22,"shimla":102.01,"Sirmaur":93.01,"Solan":80.04}
avrain = {"Chamba":71.21,"Kangra":68.05,"Kinnaur":80.55,"Kullu":79.11,"Lahul & Spiti":63.89,"Mandi":71.98,"shimla":78.19,"Sirmaur":67.35,"Solan":61.46}
sdcc = {"Chamba":11.81,"Kangra":12.39,"Kinnaur":11.79,"Kullu":11.91,"Lahul & Spiti":10.67,"Mandi":12.58,"shimla":12.87,"Sirmaur":13.58,"Solan":13.10}
avcc = {"Chamba":36.07,"Kangra":32.89,"Kinnaur":41.57,"Kullu":37.92,"Lahul & Spiti":43.19,"Mandi":34.07,"shimla":35.85,"Sirmaur":33.38,"Solan":32.19}
sdat = {"Chamba":7.11,"Kangra":7.20,"Kinnaur":5.41,"Kullu":5.41,"Lahul & Spiti":6.32,"Mandi":6.82,"shimla":6.23,"Sirmaur":6.75,"Solan":7.03}
avat = {"Chamba":19.32,"Kangra":22.34,"Kinnaur":12.69,"Kullu":12.69,"Lahul & Spiti":8.92,"Mandi":22.07,"shimla":20.56,"Sirmaur":23.99,"Solan":24.32}
sdpe = {"Chamba":1.674,"Kangra":1.626,"Kinnaur":1.23,"Kullu":1.41,"Lahul & Spiti":1.86,"Mandi":1.54,"shimla":1.39,"Sirmaur":1.46,"Solan":1.53}
avpe = {"Chamba":5.75,"Kangra":6.14,"Kinnaur":4.65,"Kullu":5.35,"Lahul & Spiti":4.13,"Mandi":6.05,"shimla":5.76,"Sirmaur":6.32,"Solan":6.37}

def club(area='Chamba',predrain = 210,predcc = 44,predtemp = 27,predpe = 8):
    predrain = ((predrain - avrain[area])/sdrain[area])
    predcc = ((predcc - avcc[area])/sdcc[area])
    predtemp = ((predtemp - avat[area])/sdat[area])
    predpe = ((predpe - avpe[area])/sdpe[area])
    
    diffacrain=((maxrain[area]-minrain[area])/5)
    diffactemp=((maxat[area]-minat[area])/5)
    diffaccc=((maxcc[area]-mincc[area])/5)
    diffacpe=((maxpe[area]-minpe[area])/5)
    ranrain=[]
    rancc=[]
    rantemp=[]
    ranpe=[]
    for i in range(0,5):
        ranrain.append(minrain[area] + i*diffacrain)
        rancc.append(mincc[area] + i*diffaccc)
        rantemp.append(minat[area] + i*diffactemp)
        ranpe.append(minpe[area] + i*diffacpe)
    for i in range(0,5):
        if ( predrain > ranrain[i] and predrain < (ranrain[i]+diffacrain)) :
            drought1 = 5-i
        else :
            continue
    for i in range(0,5):
        if ( predcc > rancc[i] and predcc < (rancc[i]+diffaccc)) :
            drought2 = 5-i
        else :
            continue
    for i in range(0,5):
        if ( predtemp > rantemp[i] and predtemp < (rantemp[i]+diffactemp)) :
            drought3 = i+1
        else :
            continue
    for i in range(0,5):
        if ( predpe > ranpe[i] and predpe < (ranpe[i]+diffacpe)) :
            drought4 = i+1
        else :
            continue
    drought = drought1+drought2+drought3+drought4
    return drought