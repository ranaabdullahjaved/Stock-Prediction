import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from keras.models import load_model
# import pandas_datareader as data

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2019, 12, 31)


st.title('STOCK TREND PREDICTION')

user_input=st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start, end)
# df=data.DataReader(user_input,'yahoo',start,end)


#Describing Data
st.subheader('DATA FROM 2010-2019')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart 100 and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


data_training= pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# Seprating 70% data for Training
data_testing= pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
# Seprating 30% data for Training
print(data_training.shape)
# For telling data in no of rows and columns
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))
data_training_array= scaler.fit_transform(data_training)


#Load My Model

model=load_model('keras_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test=np.array(x_test)
y_test=np.array(y_test)

y_predicted=model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

# final graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)

accuracy = accuracy_score(y_test, y_predicted)
st.subheader('Accuracy')