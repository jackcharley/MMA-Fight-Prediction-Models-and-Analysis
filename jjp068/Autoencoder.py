import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import MeanSquaredError
import pydot
import graphviz


"""
adapted from below source

Title: Autoencoder Feature Extraction for Classification
Author: Jason Brownlee
Date: 12/07/20
Code version: 1.0
Availability: https://machinelearningmastery.com/autoencoder-for-classification/


References:

Title: Scikit-Learn
Author: Scikit-Learn Team
Availability: https://github.com/scikit-learn/scikit-learn
Version: 0.24.2

Title: matplotlib
Author: matplotlib Team
Availability: https://github.com/matplotlib/matplotlib
Version: 3.4.2

Title: pandas
Author: pandas Team
Availability: https://github.com/pandas-dev/pandas
Version:  1.2.4

Title: pydot, pydotplus, pydot-ng
Author: pydot Team
Availability: https://github.com/pydot/pydot
Version: pydot 1.4.2, pydotplus 2.0.2, pydot-ng  2.0.0

Title: graphviz
Author: graphviz Team
Availability: https://github.com/xflr6/graphviz
Version: 0.16

Title: TensorFlow
Author: TensorFlow Team
Availability: https://github.com/tensorflow/tensorflow
Version: 2.6.0
"""

data = pd.read_csv("final_data.csv", index_col=[0])
X = data.iloc[:, :-1]
Y = data['winner']
X_train, X_test, y_train, y_test = train_test_split(MinMaxScaler().fit_transform(X), Y, test_size=0.2)
inputs = X.shape[1]
# define encoder
visible = Input(shape=(inputs,))
# encoder level 1
e = Dense(inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(inputs/2))
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss = 'mse')
# plot the autoencoder
#plot_model(model, 'autoencoder_compress.png', show_shapes=True)
# fit the autoencoder model to reconstruct input
history = model.fit(X_train, X_train, epochs=400, batch_size=100, verbose=2, validation_data=(X_test,X_test))
# plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epochs')
plt.legend()
plt.show()
# define an encoder model (without the decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
#plot_model(encoder, 'encoder_compress.png', show_shapes=True)
# save the encoder to file
encoder.save('encoder.h5')




