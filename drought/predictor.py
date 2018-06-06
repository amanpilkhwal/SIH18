import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot

import numpy
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA



def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

df_all = pd.read_csv('HIMACHAL PRADESH-DATASET(rainfall).csv', delimiter=',')
df_all.head()
location='Chamba'
df = df_all[df_all.Area == location]
df.head()

months_in_year = 12
X=df["Rainfall"].values
validation_size=24
train_size = int(len(X) -validation_size)
train, test = X[0:train_size], X[train_size:]
history = [x for x in train]
diff = difference(history, 12)
#print(history)
#print(diff)
#model = ARIMA(diff, order=(5,1,6))  
#results_MA = model.fit()
#for i in range(1,train_size):
    #print(results_MA.forecast()[i])
#final=inverse_difference(history,yhat,12,train_size)
#plt.plot(history)
#plt.plot(results_MA.fittedvalues, color='red')
#plt.title('Fitting data _ MSE: %.2f'% (((results_MA.fittedvalues-history)**2).mean()))
#plt.show()

#model_fit = ARIMAResults.load('model.pkl')
#bias = numpy.load('model_bias.npy')
# make first prediction
predictions = list()
model = ARIMA(diff, order=(6,0,5))
model_fit = model.fit()
yhat = float(model_fit.forecast()[0])
yhat = inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(test[0])

bias=-6.077092

for i in range(1,  validation_size):
	# difference data

	#diff = difference(history, months_in_year)
	# predict
	model = ARIMA(history, order=(6,0,5))
	model_fit = model.fit()
	yhat = bias+model_fit.forecast()[0]
	#yhat =inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
    # report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()