from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt

# convert series to supervised learning
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
datasetinitial = read_csv('SWaT_Dataset_Attack_v0.csv', header=0, index_col=0)
dataset_label=datasetinitial.Label
dataset=datasetinitial.drop(['Label',' MV101','P101','P102',' MV201',' P201',' P202','P203',' P204','P205','P206','MV301','MV302',' MV303','MV304','P301','P302','P401','P402','P403','P404','UV401','P501','P502','P601','P602','P603'],axis=1)

values = dataset.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 1, 1)

print(reframed.head())

# split into train and test sets
values = reframed.values
n_train_hours1 = int(0.6*(values.shape[0]))
n_train_hours2 = int(0.8*(values.shape[0]))
train = values[:n_train_hours1, :]
validate = values[n_train_hours1:n_train_hours2, :]
#validate=np.concatenate((validate,anomalyone),axis=0)
test = values[n_train_hours2:, :]

testlabels=dataset_label[n_train_hours2:-1]
# split into input and outputs
train_X, train_y = train[:, :25], train[:, 25:]
validate_X, validate_y = validate[:, :25], validate[:, 25:]
test_X, test_y = test[:, :25], test[:, 25:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
validate_X = validate_X.reshape((validate_X.shape[0], 1, validate_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(25))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(validate_X, validate_y), verbose=2,
                    shuffle=False)
# plot history

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
print(yhat.shape)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 25:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
print(inv_yhat.shape)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 25))
inv_y = concatenate((test_y, test_X[:, 25:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)

# calculate RMSE
ypred=np.array([])
ypredf=np.array([])
ypredfinal=np.zeros(inv_yhat.shape[0])
rmse=np.zeros(inv_yhat.shape[1])
for i in range(inv_yhat.shape[1]):
	rmse[i] = sqrt(mean_squared_error(inv_y[:,i], inv_yhat[:,i]))

ypred=np.zeros([inv_yhat.shape[0],inv_yhat.shape[1]])
ypredf=np.zeros([inv_yhat.shape[0],inv_yhat.shape[1]])

for i in range(inv_yhat.shape[1]):
	ypred[:,i] =np.subtract(inv_yhat[:,i],inv_y[:,i])
	ypredf[:,i]=np.abs(ypred[:,i])


for i in range(inv_yhat.shape[0]):
	for j in range(inv_yhat.shape[1]):
		if ypredf[i,j]<rmse[j]:
			ypredfinal[i]=1
			break
		else:
			ypredfinal[i]=0
plt.plot(yhat,'r')
plt.plot(test_y,'b')
plt.show()

print(f1_score(testlabels,ypredfinal,average="binary"))
print(accuracy_score(testlabels,ypredfinal))
print('Test RMSE: %.3f' % rmse[0])
