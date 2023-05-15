import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


# generate data 
x = [[[(i+j/100)] for i in range(5)] for j in range(100)]
y = [(i+5)/100 for i in range(100)]

#((100, 5, 1), (100,))
x = np.array(x)
y = np.array(y)
x.shape, y.shape

#((80, 5, 1), (20, 5, 1), (80,), (20,))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


model = Sequential()
model.add(SimpleRNN(100, batch_input_shape=(None, 5,1), return_sequences=True))
model.add(SimpleRNN((1),return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
model.summary()

# shows the following o/p
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn_9 (SimpleRNN)    (None, 5, 100)            10200     
                                                                 
#  simple_rnn_10 (SimpleRNN)   (None, 1)                 102       
                                                                 
#  dense_5 (Dense)             (None, 1)                 2         
                                                                 
# =================================================================
# Total params: 10,304
# Trainable params: 10,304
# Non-trainable params: 0
# ______________________________


history = model.fit(x, y, epochs=400, validation_data=(x_test, y_test), verbose=0)
result = model.predict(x_test)
result.shape #(20, 1)

# history.history.keys() 
# history.history["accuracy"]


import matplotlib.pyplot as plt
plt.scatter(range(20), result, c='r')
plt.scatter(range(20), y_test, c='g')
plt.show();
