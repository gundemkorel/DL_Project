#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScale


import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
#%%
file_path = r'C:\Users\PC\Desktop\data_set _for_project.xls'  # replace with your file path

df = pd.read_excel(file_path, sheet_name='FRED Graph')
#%%
df
#%%
df1 =df.copy()
df1['year'] = df1['observation_date'].apply(lambda x: x.year)
df1['quarter'] = df1['observation_date'].apply(lambda x: x.quarter)
df1['month'] = df1['observation_date'].apply(lambda x: x.month)
#%%
fig = px.box(df1[12:], x="month", y="Consumer_Price", points = "all", template = "presentation",)
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',))

fig = px.box(df1[12:], x="quarter", y="Consumer_Price", points = "all", template = "presentation")

fig.show()
#%%
fig = px.bar(
    data_frame=df1.groupby(['month']).std().reset_index(),
    x="month",
    y="Consumer_Price", text="Consumer_Price"
).update_traces(texttemplate='%{text:0.3f}', textposition='outside').update_xaxes(nticks=13)
fig.show()

fig = px.bar(
    data_frame=df1.groupby(['quarter']).std().reset_index(),
    x="quarter",
    y="Consumer_Price", text="Consumer_Price").update_traces(texttemplate='%{text:0.3f}', textposition='outside').update_xaxes(nticks=5)
fig.show()
#%%
df = df.set_index('observation_date')

df['Consumer_Price'].plot()
fig = seasonal_decompose(df['Consumer_Price'], model='additive').plot()
df_cpi = df.copy()
#%%
split_point = len(df_cpi) - 12
train, test = df_cpi[0:split_point], df_cpi[split_point:]
print('Training dataset: %d, Test dataset: %d' % (len(train), len(test)))
train['Consumer_Price'].plot()
test['Consumer_Price'].plot()
#%%
def adf_test(df):
    result = adfuller(df.values)
    if result[1] > 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")
adf_test(train['Consumer_Price'])
#%%
diff = train['Consumer_Price'].diff()
plt.plot(diff)
plt.show()
diff = diff.dropna()
#%%
adf_test(diff)

#%%
plot_pacf(diff.values)
#%%
plot_acf(diff.values)
#%%
arima_model = ARIMA(np.log(train['Consumer_Price']), order = (1,1,1),freq=train.index.inferred_freq)

arima_fit = arima_model.fit()
arima_fit.summary()
#%%
forecast = arima_fit.forecast(steps=12)
forecast = np.exp(forecast)

plt.plot(forecast, color = 'red')
plt.plot(test['Consumer_Price'])
#%%
forecast
#%%
mse = mean_squared_error(test['Consumer_Price'].values, forecast[:12])
print('MSE: ', mse)

#%%
ccpi = df['Consumer_Price'].values

#%%
values = [0.54440000000, 0.61140000000, 0.69970000000, 0.73500000000, 0.78620000000,
          0.79600000000, 0.80210000000, 0.83450000000, 0.85510000000, 0.8439, 0.6427, 0.5768]

# Convert the list to a numpy array
arr = np.array(values)

arr = pd.DataFrame(arr, index= test.index)
#%%
arima_model = ARIMA(np.log(df_cpi['Consumer_Price']), order = (1,1,1),freq=test.index.inferred_freq)

arima_fit = arima_model.fit()

forecast = arima_fit.forecast(steps=12)
forecast = np.exp(forecast)
plt.plot(forecast, color = 'red',label='forecasted')




#%%
plt.plot(pd.date_range('2022-01-01',periods=12,freq='M'),arr, color='blue',label='actual')
#%%
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(ccpi.reshape(-1,1))
#%%
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
#%%
n_steps_in = 12

train, test = dataset[0:313], dataset[313:len(dataset),:]

trainX, trainY = split_sequence(train, n_steps_in)
testX, testY = split_sequence(test, n_steps_in)
#%%

#%%
n_features = trainX.shape[2]

uni_model = Sequential()

# Adding the LSTM layer
uni_model.add(LSTM(64, input_shape=(trainX.shape[1], n_features)))

# Adding the output layer
uni_model.add(Dense(1))

uni_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = 'mean_squared_error', metrics=['mean_absolute_error'])

fit = uni_model.fit(trainX,
          trainY, validation_data = (testX, testY),
          epochs = 100, batch_size=1,
          verbose = 0)


# Check for overfitting
plt.plot(fit.history['loss'], label = 'training', color = 'Blue')
plt.plot(fit.history['val_loss'], label = 'validation', color = 'Red')
plt.legend()
plt.show
#%%
trainPredict = uni_model.predict(trainX)
testPredict = uni_model.predict(testX)

Ytrain_hat = scaler.inverse_transform(trainPredict)
Ytrain_actual = scaler.inverse_transform(trainY)
Ytest_hat = scaler.inverse_transform(testPredict)
Ytest_actual = scaler.inverse_transform(testY)
#%%
trainScore = mean_squared_error(Ytrain_actual, Ytrain_hat[:,0])
print('Train Score: %.5f MSE' % (trainScore))
testScore = mean_squared_error(Ytest_actual, Ytest_hat[:,0])
print('Test Score: %.5f MSE' % (testScore))

model_error = Ytest_actual - Ytest_hat[:,0]
print('Mean Model Error: ', model_error.mean())
#%%
observed = df_cpi.loc['2021-02-01':'2022-01-01',['Consumer_Price']]
observed.plot(color = 'SteelBlue', title = 'Actual', legend = False)


predicted = pd.DataFrame(Ytest_hat, index=pd.date_range('2021-02-01',periods=12,freq='M'))
predicted.plot(color = 'Firebrick', title = 'Forecasted', legend = False)
plt.show()
#%%
x_input = np.array(dataset[-12:])
x_input = x_input.reshape((1, n_steps_in, n_features))

forecast_normalized = uni_model.predict(x_input)

forecast = scaler.inverse_transform(forecast_normalized)
forecast
#%%
#out of sample values
values = [0.54440000000, 0.61140000000, 0.69970000000, 0.73500000000, 0.78620000000,
          0.79600000000, 0.80210000000, 0.83450000000, 0.85510000000, 0.8439, 0.6427, 0.5768]

# Convert the list to a numpy array
arr = np.array(values)

arr = arr.reshape(-1,1)

#%%
kore =scaler.fit_transform(np.vstack((scaler.inverse_transform(np.array(dataset[-12:])),arr)))
#%%
a, b = split_sequence(kore, n_steps_in)
#%%
out_of_sample_forecast = []
for i in range(12):
    x_input= a[i]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    forecast_normalized = uni_model.predict(x_input)

    forecast = scaler.inverse_transform(forecast_normalized)
    out_of_sample_forecast.append(forecast)


#%%
b = scaler.inverse_transform(b)
out_of_sample_forecast = np.concatenate(out_of_sample_forecast, axis=0)
#%%


plt.plot(pd.date_range('2022-01-01',periods=12,freq='M'),out_of_sample_forecast, color='red',label='forecasted')
plt.plot(pd.date_range('2022-01-01',periods=12,freq='M'),b, color='blue',label='realization')
plt.title('Out of Sample Test')
plt.legend()
plt.show()

#%%
oos = mean_squared_error(out_of_sample_forecast, b)
print('Out of Sample Score Score: %.5f MSE' % (oos))

#%%
monthly_df_stationary = df.copy()
for indi in monthly_df_stationary:
    print('ADF Test: ', indi)
    adf_test(monthly_df_stationary[[indi]])
#%%
monthly_df_stationary = df_cpi.diff().dropna()
for indi in monthly_df_stationary:
    print('ADF Test: ', indi)
    adf_test(monthly_df_stationary[[indi]])

#%%
monthly_df_stationary[['M2']] = monthly_df_stationary[['M2']].diff().dropna()
monthly_df_stationary[['Real_ef_Ex']] = monthly_df_stationary[['Real_ef_Ex']].diff().dropna()

#%%
monthly_df_stationary = monthly_df_stationary.dropna()

for indi in monthly_df_stationary:
    print('ADF Test: ', indi)
    adf_test(monthly_df_stationary[[indi]])
#%%
maxlag=12
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(monthly_df_stationary, variables = monthly_df_stationary.columns)
#%%
feat_df = df.drop(['Ind_Prod', 'registered_unemp','Buisness_ten'], axis = 1)
#%%
scaled = scaler.fit_transform(feat_df)
#%%
scaled_df = pd.DataFrame(scaled, columns=feat_df.columns, index=feat_df.index)
scaled_df.head(5)
#%%
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        #find end of pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1

        #check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break

        #gather input and output
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)
#%%
in_cpi = np.array(scaled_df.loc['1994-01-01':'2020-02-01', ['Consumer_Price']])
in_ur = np.array(scaled_df.loc['1994-01-01':'2020-02-01', ['interest_rate']])
in_m2  = np.array(scaled_df.loc['1994-01-01':'2020-02-01', ['M2']])
in_ex = np.array(scaled_df.loc['1994-01-01':'2020-02-01', ['Real_ef_Ex']])



test_cpi = np.array(scaled_df.loc['2020-03-01':'2022-01-01', ['Consumer_Price']])
test_ur = np.array(scaled_df.loc['2020-03-01':'2022-01-01', ['interest_rate']])
test_m2 = np.array(scaled_df.loc['2020-03-01':'2022-01-01', ['M2']])
test_ex  = np.array(scaled_df.loc['2020-03-01':'2022-01-01', ['Real_ef_Ex']])


trainoutput_cpi = in_cpi
testoutput_cpi = test_cpi
#%%
in_cpi = in_cpi.reshape((len(in_cpi), 1))
in_ur = in_ur.reshape((len(in_ur), 1))
in_m2  = in_m2.reshape((len(in_m2), 1))
in_ex = in_ex.reshape((len(in_ex), 1))


test_cpi = test_cpi.reshape((len(test_cpi), 1))
test_ur = test_ur.reshape((len(test_ur), 1))
test_m2  = test_m2.reshape((len(test_m2), 1))
test_ex = test_ex.reshape((len(test_ex), 1))

trainoutput_cpi = trainoutput_cpi.reshape((len(trainoutput_cpi), 1))
testoutput_cpi = testoutput_cpi.reshape((len(testoutput_cpi), 1))
#%%
trainset = np.hstack((in_cpi, in_ur, in_m2, in_ex, trainoutput_cpi))
testset = np.hstack((test_cpi, test_ur, test_m2, test_ex, testoutput_cpi))

n_steps_in = 12
n_steps_out = 1

trainX, trainy = split_sequences(trainset, n_steps_in, n_steps_out)

testX, testy = split_sequences(testset, n_steps_in, n_steps_out)

trainX.shape, trainy.shape
#%%
n_features = trainX.shape[2]

multi_model = Sequential()

# Adding the LSTM layer and dropout regularizaiton
multi_model.add(LSTM(100, return_sequences = True, input_shape=(n_steps_in, n_features)))
multi_model.add(LSTM(100))
multi_model.add(Dropout(0.2))

# Adding output layer
multi_model.add(Dense(n_steps_out))

multi_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = 'mean_squared_error')

earlystop = EarlyStopping(monitor = 'val_loss', patience =50,
                  mode = 'min',
                  verbose = 0)

fit = multi_model.fit(trainX,
          trainy, validation_data = (testX, testy),
          epochs = 500, verbose=0, callbacks = [earlystop])


# Check for overfitting
plt.plot(fit.history['loss'], label = 'training', color = 'Blue')
plt.plot(fit.history['val_loss'], label = 'validation', color = 'Red')
plt.legend()
plt.show()
#%%
def feature_importance(model, g):
    random_ind = np.random.choice(g.shape[0], 100, replace=False) # Randomly generate 100 numbers arange(218)
    x = g[random_ind] #  Take 100 random sample from training set
    orig_out = model.predict(x)
    for i in range(4):  # iterate over the 4 features
        new_x = x.copy()
        perturbation_in = np.random.normal(0.0, 0.7, size=new_x.shape[:2]) # Draw random samples from normal distribution with sd = 0.7, this value is arbitary and would not affect the order of effect as its just introducing noise.
        new_x[:, :, i] = new_x[:, :, i] + perturbation_in
        perturbed_out = model.predict(new_x)
        effect = ((orig_out - perturbed_out) ** 2).mean() ** 0.5
        print(f'Variable {i+1}, Perturbation Effect: {effect:.3f}')

feature_importance(multi_model,trainX)
#%%
testPredict = multi_model.predict(testX)
#%%
testX = testX.reshape((testX.shape[0], testX.shape[2]*testX.shape[1]))
#%%
testX
#%%
# Invert scaling for Predicted
testY_hat = np.concatenate((testX[:, 1:4], testPredict), axis=1)
testY_hat = scaler.inverse_transform(testY_hat)

testY_hat = testY_hat[:,3]

# Invert scaling for Actual
testY_actual = np.concatenate((testX[:,1:4], testy), axis=1)
testY_actual = scaler.inverse_transform(testY_actual)

testY_actual = testY_actual[:,3]
#%%
mse = mean_squared_error(testY_actual, testY_hat)

print('Test MSE: %.5f' % mse)

#%%
observed = df_cpi.loc['2021-02-01':'2022-01-01',['Consumer_Price']]
observed.plot(color = 'SteelBlue', title = 'Actual', legend = False)
plt.show()

predicted = pd.DataFrame(testY_hat/10, index=pd.date_range('2021-02-01',periods=12,freq='M'))
predicted.plot(color = 'Firebrick', title = 'Forecasted', legend = False)
plt.show()
#%%
x_input = np.array(scaled[-12:])
x_input = x_input.reshape((1, n_steps_in, n_features))

forecast_normalized = multi_model.predict(x_input)

# Manually inverse Min-max normalization
max_cpi = df_cpi['Consumer_Price'].max()
min_cpi = df_cpi['Consumer_Price'].min()
forecast =  max_cpi-forecast_normalized[0][0]/(max_cpi-min_cpi)

forecast
#%%
file_path = r'C:\Users\PC\Desktop\data_set _for_project.xls'  # replace with your file path

df = pd.read_excel(file_path, sheet_name='Sheet1')
df.set_index('observation_date',inplace=True)

df
#%%
df.drop(['Ind_Prod','registered_unemp','Buisness_ten'],axis=1,inplace=True)
#%%
df1 = df
df1
#%%
second_input = np.concatenate((scaler.inverse_transform(scaled[-12:]),df1), axis=0)
second_input = scaler.fit_transform(second_input)
#%%
new_column = second_input[:, 0][:, np.newaxis]
second_input = np.concatenate((second_input, new_column), axis=1)
second_input
#%%
ece, ecem =split_sequences(second_input, n_steps_in, n_steps_out)
#%%
out_of_sample_forecast = []
for i in range(12):
    x_inputs= ece[i]
    x_inputs = x_inputs.reshape((1, n_steps_in, n_features))
    forecast = multi_model.predict(x_inputs)
    forecast =  max_cpi-forecast[0][0]/(max_cpi-min_cpi)


    out_of_sample_forecast.append(forecast)
#%%
plt.plot(pd.date_range('2022-01-01', periods=12, freq='M'), out_of_sample_forecast, color='red', label='forecasted')
plt.plot(pd.date_range('2022-01-01', periods=12, freq='M'), b, color='blue', label='realization')
plt.title('Out of Sample Test')
plt.legend()
plt.show()
