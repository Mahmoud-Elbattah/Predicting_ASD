import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


np.random.seed(123)
my_data = np.genfromtxt('F:\\ASDDataset.csv', delimiter=',')


features = my_data[1:,0:my_data.shape[1]-1]
labels = my_data[1:,features.shape[1]].astype(dtype='int')

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30, random_state=42)


X_train = X_train.reshape(X_train.shape[0],  200, 200,1)
X_test = X_test.reshape(X_test.shape[0],  200, 200,1)

Y_train = np_utils.to_categorical(Y_train, 2)
Y_test = np_utils.to_categorical(Y_test, 2)

# 7. Define model architecture
model = Sequential()
model.add(Convolution2D(32, 2, 2, input_shape=(200, 200, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, nb_epoch=10, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
AUC = roc_auc_score(Y_test, model.predict(X_test))
print(AUC)