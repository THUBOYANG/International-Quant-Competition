
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,Activation

#读取数据：train 4876，test 863
dataset_train = pd.read_csv('data_train.csv')
dataset_test = pd.read_csv('data_test.csv')

training_set = dataset_train.iloc[:, 1:].values
print(training_set.shape)
print(training_set[:,-4])


# In[206]:



sc_x = MinMaxScaler(feature_range = (0, 1))
sc_y = MinMaxScaler(feature_range = (0, 1))
training_set_scaled_x = sc_x.fit_transform(np.hstack((training_set[:,:-5],training_set[:,-3:])))
training_set_scaled_y = sc_y.fit_transform(training_set[:,[-4]])
training_set_scaled = np.hstack((training_set_scaled_x,training_set_scaled_y))
print(training_set_scaled.shape)
plt.plot(training_set_scaled[:,-1]) #plot predict after EMD


# In[207]:


X_train, y_train = training_set_scaled[:,:-1], training_set_scaled[:,[-1]]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train)
print(y_train)
plt.plot(y_train)


# In[157]:





# In[210]:


u=128
d=0.2
num_stack_layers = 5

regressor = Sequential()

regressor.add(LSTM(units = u, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(d))
# for i in range(num_stack_layers-2):
#     regressor.add(LSTM(units = u, return_sequences = True))
#     regressor.add(Dropout(d))
regressor.add(LSTM(units = u))
regressor.add(Dropout(d))

regressor.add(Dense(1,activation = 'linear'))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 15, batch_size = 32)


# In[185]:



print(dataset_test.shape)


# In[186]:


# N = len(dataset_test_new)
# volatility = []
# ma3d = []
# ma10d = []
# for i in range(10,N):
#     volatility.append(np.std(dataset_test_new[(i-10):i,-1]))
#     ma3d.append(np.mean(dataset_test_new[(i-3):i,-1]))
#     ma10d.append(np.mean(dataset_test_new[(i-10):i,-1]))
# dataset_test_new = pd.DataFrame(dataset_test_new[10:,:],columns=dataset_test.columns[1:],index = dataset_test['date'])
# dataset_test_new['volatility10d'] = np.array(volatility)
# dataset_test_new['ma3d'] = np.array(ma3d)
# dataset_test_new['ma10d'] = np.array(ma10d)
# dataset_test_new.head(3)


# In[187]:


test_set = dataset_test.iloc[:,1:].values
print(test_set.shape)
X_test = np.hstack((test_set[:,:-4],test_set[:,-3:]))
print(X_test.shape)


# In[188]:


X_test_scaled = sc_x.transform(X_test)
print(X_test_scaled)


# In[189]:


X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0],X_test_scaled.shape[1],1))
#print(X_test_scaled)
y_predict_scaled = regressor.predict(X_test_scaled)
print(y_predict_scaled)


# In[182]:


#print(y_predict_scaled.shape)
# y_predict = sc_y.inverse_transform(y_predict_scaled)
# print(y_predict)


# In[191]:


plt.plot(y_predict, color = 'green', label = 'Predicted high freq terms')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[192]:


y_train_predict = regressor.predict(X_train)
print(y_train_predict)
plt.plot(y_train, color = 'blue', label = 'Real Price')
plt.plot(y_train_predict, color = 'red', label = 'Predicted Price')
plt.title('Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

