# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IPNCT9_sEwdgBIr4WzN7tgQS5z-cUG_V
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_india = pd.read_csv("/content/covid_19_india.csv")
df_india.head()

df_india.info()

df_global = pd.read_csv("/content/Global covid data.csv")
df_global.head()

df_global.info()

df_statewise = pd.read_csv("/content/covid_vaccine_statewise.csv")
df_statewise.head()

df_statewise.info()



### df_statewise used

def data_information(data):
  print("Information:\n",data.info())
  print("Describtion:\n",data.describe())
  null = pd.DataFrame(data.isnull().sum())
  null["precentage"] = (data.isnull().sum()/len(data))*100
  return null


null = data_information(df_india)

null.style.background_gradient()

df_india.duplicated().sum()

numerical_data = [col for col in df_india.columns if df_india[col].dtypes!="O"]

numerical_data

## statistcal analysis
## Graphical

df_india.plot.box(figsize=(15,7))
plt.xlabel("Columns")
plt.title("Boxplot To Anomaly Detection")
plt.show()


for i in numerical_data:
  df_india[i].plot.hist(figsize=(15,7))
  plt.xlabel(i)
  plt.title("histogram")
  plt.show()

df_india.corr().style.background_gradient()

### FeatureEngineering

df_india

import datetime

df_india["day"] = pd.to_datetime(df_india["Date"]).dt.day
df_india["year"] = pd.to_datetime(df_india["Date"]).dt.year
df_india["month"] = pd.to_datetime(df_india["Date"]).dt.month

new = df_india["Time"].str.split(" ",expand=True)

new.columns = ["time","Day/Night"]
new.head()

df1 = pd.concat([df_india,new],axis=1)

period = []
for i in df1["Day/Night"]:
  if i == "PM":
    period.append("Day")
  else:
    period.append("Night")

df1["Day/Night"] = period

pd.DataFrame(period).value_counts()

df1["Time"] = df1["time"].apply(lambda x: int(x[0]))

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()




df1["State/UnionTerritory"] = label.fit_transform(df["State/UnionTerritory"])

labels = {}
labels["State"] = label

label = LabelEncoder()




df1["Day/Night"] = label.fit_transform(df1["Day/Night"])
labels["Day/Night"] = label

labels

X = df1.drop(["time","Date","Confirmed"],axis=1)
y = df1["Confirmed"]

Test = df1.iloc[:int(.2*len(df1)),:]
Train  = df1.iloc[int(.2*len(df1)):,:]

from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=.2,random_state=2,shuffle=True)

X_train.shape,X_valid.shape

X_train["ConfirmedForeignNational"]=X_train["ConfirmedForeignNational"].replace("-",0).astype(int)
X_train["ConfirmedIndianNational"]=X_train["ConfirmedIndianNational"].replace("-",0).astype(int)

X_valid["ConfirmedForeignNational"]=X_valid["ConfirmedForeignNational"].replace("-",0).astype(int)
X_valid["ConfirmedIndianNational"]=X_valid["ConfirmedIndianNational"].replace("-",0).astype(int)

y_train.dtypes

import statsmodels.api as sm

features = sm.add_constant(X_train)
model_stats = sm.OLS(exog = features.drop(["Deaths"],axis=1),endog = y_train).fit()
model_stats.summary()

##

import tensorflow as tf

from tensorflow.keras.layers import Dense,LSTM,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
test_X = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))

y_train.values

import matplotlib.pyplot as plt
# def network
model = Sequential()
model.add(Dense(128,activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dense(1))
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=.01),loss="mae")
model.summary()
# fit network
history = model.fit(train_X, y_train, epochs=500, batch_size=512, validation_data=(test_X,y_valid), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

## evaluate the model and calculate Train RMSE
train_yhat = model.predict(train_X)
train_yhat

from sklearn.metrics import mean_squared_error,mean_absolute_error

print("MSE:",mean_squared_error(train_yhat,y_train.values))
print("MAE:",mean_absolute_error(train_yhat,y_train.values))

import joblib
joblib.dump(model,"Model")