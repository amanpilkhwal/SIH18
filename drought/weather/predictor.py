import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import copy

def read_data(location):
	df = pd.read_csv('HIMACHAL PRADESH-DATASET(rainfall).csv', delimiter=',')
	location_df = df[df.Area == location]
	return location_df

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def get_datasets(df):
	X=df["Rainfall"].values
	validation_size=24
	train_size = int(len(X) -validation_size)
	return X[0:train_size], X[train_size:]
	# return cross_validation.train_test_split(df["Rainfall"], df["Month"], test_size=0.25, random_state=10,000)

def train_model(diff):
	model = ARIMA(diff, order=(6,0,5))
	return model.fit()

def test_model():
	pass

def predict(test, history, predictions, validation_size=24, bias=-6.077092):
	for i in range(1,  validation_size):
		model = ARIMA(history, order=(6,0,5))
		model_fit = model.fit()
		yhat = bias+model_fit.forecast()[0]
		#yhat =inverse_difference(history, yhat, months_in_year)
		predictions.append(yhat)
		# observation
		obs = test[i]
		history.append(obs)
		print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
	return history, predictions


def main(location='Chamba'):
	df = read_data(location)
	print(df)
	train, test= get_datasets(df)
	history = [x for x in train]
	diff = difference(history, 12)
	months_in_year = 12
	model_fit = train_model(diff)
	yhat = float(model_fit.forecast()[0])
	yhat = inverse_difference(history, yhat, months_in_year)
	predictions = []
	predictions.append(yhat)
	history.append(test[0])

	history, predictions = predict(test, history, predictions)
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	print('RMSE: %.3f' % rmse)
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.show()
main()

