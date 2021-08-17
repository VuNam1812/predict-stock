# # This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import random
import datetime as dt
import matplotlib.pyplot as plt

from stock_pred import *
from PIL import Image
from bs4 import BeautifulSoup
from numpy.lib.function_base import append


# Page layout
# Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title
image = Image.open('images_main.jpg')

st.image(image, width=500)

st.title('Stock Price Predictions WebsiteApp')
st.write("""
Ứng dụng này truy xuất giá chứng khoán cho 100 loại chứng khoán hàng đầu từ **Yahoo finance**!

""")
#---------------------------------#
# About
expander_bar = st.expander("Về ứng dụng")
expander_bar.markdown("""
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
* **Data source:** [Yahoo Finance](https://finance.yahoo.com/).
""")


#---------------------------------#
# Page layout (continued)
# Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((1, 1))

#---------------------------------#
# Sidebar + Main panel
col1.header('Đầu vào các yêu cầu')

# Sidebar
ticket_selected = st.sidebar.selectbox(
    'Dữ liệu chứng khoán', ['AMZN', 'AAPL', 'GOOG', 'FB', 'TSLA'])
currency_algorithms = col1.selectbox('Thuật toán dự đoán', ('LSTM', 'RNN'))
currency_type_uint = col1.selectbox(
    'Kiểu dự đoán', ('Close', 'Price rate of change'))
currency_interval = col1.selectbox(
    'Khoảng dự đoán', ('15m', '30m', '1h', '1d'))
currency_dateBegin = col1.date_input('Ngày bắt đầu', dt.datetime(2021,8, 1))
currency_dateEnd = col1.date_input('Ngày Kết thúc', dt.datetime(2021, 8, 15))

scale = MinMaxScaler(feature_range=(0, 1))

model_selected = train_init_model(
    ticket_selected, currency_algorithms, currency_type_uint, scale)
test_inputs = create_data(
    scale, ticket_selected, currency_interval, currency_dateBegin, currency_dateEnd)

predictions = prediction_price(scale, test_inputs, model_selected)
predictions = pd.DataFrame(data=predictions, columns=['Close'])
valid = load_data_predics(
    ticket_selected, currency_interval, currency_dateBegin, currency_dateEnd)
valid_total = load_data_pre_predics(ticket_selected, currency_interval, currency_dateBegin, currency_dateEnd)

calc_price_rate(valid_total)
before_value_first=valid_total['Close'][len(valid_total) - len(valid)-1]
# get value before first predictions
calc_price_rate(predictions, before_value_first)

valid_total=valid_total.filter(
    [convert_type_prediction.get(currency_type_uint)])
valid=valid_total[len(valid_total) - len(valid):]
valid['Prediction']=np.array(
    predictions[convert_type_prediction.get(currency_type_uint)])
st.markdown("""
**Dữ liệu dự đoán và thực tế**
""")
st.write(valid)

draw_chart(valid, currency_type_uint)
