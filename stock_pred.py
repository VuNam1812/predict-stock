import streamlit as st
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas_datareader as web
import pandas as pd
import numpy as np
from streamlit.caching import cache
import yfinance as yf
import os.path
import plotly.express as px

from plotly import graph_objs as go
from keras.layers import LSTM, SimpleRNN, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from os import path
from keras import models
from keras.saving.save import load_model
from numpy.core.fromnumeric import reshape
st.set_option('deprecation.showPyplotGlobalUse', False)


# prepare data

convert_timeData = {
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '1d': 86400,
    '7d': 604800,
    '1mo': 2629746,
    '6mo': 15778476,
    '1y': 31556952,
    '5y': 157784760,
    '8y': 315569520
}

convert_interval_time = {
    '15m': ['1d', '7d', '1mo'],
    "30m": ['1d', '7d', '1mo'],
    "1h": ['1d', '7d', '1mo', '6mo', '1y'],
    "1d": ['1mo', '6mo', '1y', '5y', '8y'],
}

convert_type_prediction = {
    'Close': 'Close',
    'Price rate of change': "ROC"
}


@st.cache(allow_output_mutation=True)
def load_data_predics(ticket, interval, startDate, endDate):
    data = yf.download(
        tickers=str(ticket),
        start=startDate,
        end=endDate,
        interval=str(interval),
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None
    )

    return data


@st.cache(allow_output_mutation=True)
def load_data_pre_predics(ticket, interval, startDate, endDate):
    startDateNew = dt.datetime.fromtimestamp(time.mktime(
        startDate.timetuple()) - (int(convert_timeData.get(interval) * 1000)))
    data = yf.download(
        tickers=str(ticket),
        start=startDateNew,
        end=endDate,
        interval=str(interval),
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None
    )

    return data


def load_data_train(scale, selected_type, ticket, interval, startDate, endDate):
    # create dataframe with only 'Close' column
    # convert dataframe to numpy array
    data = load_data_predics(str(ticket), str(interval), startDate, endDate).filter(
        [convert_type_prediction.get(selected_type)]).values

    # scale data
    scaled_data = scale.fit_transform(data)
    return scaled_data


def prepare_data_train(train_data, prediction=80):
    data = []

    # copy train_data into data
    for i in range(prediction, len(train_data)):
        data.append(train_data[i-prediction: i, 0])

    # convert data to numpy array
    data = np.array(data)
    # convert shape to LSTM (LSTM require shape 3D)
    # data_train_modal.shape => [rows.length, cols.length]

    data = np.reshape(data, (data.shape[0], data.shape[1], 1))

    return data


def prepare_result_data_train(train_data, prediction=80):
    data = []

    # copy train_data into data
    for i in range(prediction, len(train_data)):
        data.append(train_data[i, 0])

    # convert data to numpy array
    data = np.array(data)
    data = np.reshape(data, (data.shape[0], 1))
    return data

# build & train model LSTM


def prepare_model_lstm(data_train, result_train, loop=100):
    result_model = Sequential()
    result_model.add(LSTM(50, return_sequences=True,
                          input_shape=(data_train.shape[1], 1)))
    result_model.add(Dropout(0.2))
    result_model.add(LSTM(50, return_sequences=False))
    result_model.add(Dropout(0.2))
    result_model.add(Dense(1))
    # compile model
    result_model.compile(
        optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    if path.exists('lstm_model_Close.h5'):
        result_model.load_weights('lstm_model_Close.h5')
    else:
        result_model.fit(data_train, result_train,
                         batch_size=64, epochs=loop)

        result_model.save('lstm_model_Close.h5')

    return result_model


def prepare_model_rnn(data_train, result_train, loop=100):
    result_model = Sequential()
    result_model.add(SimpleRNN(50, input_shape=(data_train.shape[1], 1)))
    result_model.add(Dropout(0.2))
    result_model.add(Dense(1))
    # compile model
    result_model.compile(
        optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    if path.exists('rnn_model_Close.h5'):
        result_model.load_weights('rnn_model_Close.h5')
    else:
        st.write(result_train.shape)
        st.write(data_train.shape)
        result_model.fit(data_train, result_train,
                         batch_size=64, epochs=loop)

        result_model.save('rnn_model_Close.h5')

    return result_model
# create test data


def create_data(scale, ticket, interval, startDate, endDate, prediction=80):
    # download data
    # filter and convert dataframe to numpy array
    data = load_data_predics(str(ticket), str(interval), startDate, endDate)
    data_total = load_data_pre_predics(
        str(ticket), str(interval), startDate, endDate)
    calc_price_rate(data_total)

    data_total = data_total.filter(['Close'])

    # get 60 first rows of test_data
    test_inputs = data_total[len(data_total) - len(data) - prediction:].values

    # scale data
    test_inputs = test_inputs.reshape(-1, 1)
    test_inputs = scale.transform(test_inputs)

    return test_inputs
# prediction data


def prediction_price(scale, test_inputs, model, prediction=80):

    data_test = prepare_data_train(test_inputs, prediction)

    # get value predicted from model
    predictions = model.predict(data_test)
    # convert to real scale
    predictions = scale.inverse_transform(predictions)

    return predictions
# function draw chart


def draw_chart(valid, selected_type):
    fig = go.Figure(layout=go.Layout(height=600, width=900))
    if convert_type_prediction.get(selected_type) == 'Close':
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid['Close'], name='Stock valid close', mode="lines"))
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid['Prediction'], name='Stock prediction close', mode="lines"))
        fig.layout.update(title_text='Biểu đồ dự đoán giá đóng')

    else:
        fig.add_trace(go.Bar(name='Valid', x=valid.index, y=valid['ROC']))
        fig.add_trace(go.Bar(name='Prediction',
                      x=valid.index, y=valid['Prediction']))

        # Change the bar mode
        fig.update_layout(barmode='group')
        fig.layout.update(title_text='Biểu đồ phần trăm tỷ lệ thay đổi giá')
    st.plotly_chart(fig)


@st.cache(allow_output_mutation=True)
def download_data_init(ticket):
    data = yf.download(
        tickers=str(ticket),
        period='max',
        interval='1d',
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None,
    )
    return data


def train_init_model(ticket, algorithm, selected_type, scale):
    train_data = download_data_init(ticket)
    calc_price_rate(train_data)
    train_data = train_data.filter(['Close']).values
    train_data_scaled = scale.fit_transform(train_data)

    data_train_model = prepare_data_train(train_data_scaled)
    result_train_model = prepare_result_data_train(train_data_scaled)
    model = Sequential()
    if algorithm == 'LSTM':
        model = prepare_model_lstm(
            data_train_model, result_train_model)
    else:
        model = prepare_model_rnn(
            data_train_model, result_train_model)
    return model


def calc_price_rate(data, beforeFistValue=0):
    price_change = []
    if beforeFistValue == 0:
        price_change.append(0)
    else:
        percent_change = ((data['Close'][0] / beforeFistValue) - 1) * 100
        price_change.append(percent_change)

    for i in range(1, len(data['Close'])):

        percent_change = ((data['Close'][i] / data['Close'][i-1]) - 1) * 100
        price_change.append(percent_change)

    data['ROC'] = price_change


prediction_days = 80
scale = MinMaxScaler(feature_range=(0, 1))
