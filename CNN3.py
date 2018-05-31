import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D,MaxPooling1D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(123)
my_data = np.genfromtxt('D:\\Dataset_transformed.csv', delimiter=',')

X = my_data[1:,0:my_data.shape[1]-1]
Y = my_data[1:,X.shape[1]].astype(dtype='int')

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scoresAUC = []
for train, test in kfold.split(X, Y):
    X_train = X[train]
    Y_train = Y[train]
    X_test = X[test]
    Y_test = Y[test]

    X_train = X_train.reshape(X_train.shape[0],  50, 1)
    X_test = X_test.reshape(X_test.shape[0], 50, 1)
    model = Sequential()
    model.add(Convolution1D(nb_filter=25, filter_length=10, input_shape=(50, 1)))
    model.add(Activation('relu'))
    #model.add(MaxPooling1D())

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=16, nb_epoch=100, verbose=0)
    #Evaluate model on test data
    #score = model.evaluate(X_test, Y_test, verbose=0)

    AUC = roc_auc_score(Y_test, model.predict(X_test))
    scoresAUC.append(AUC)
    print(AUC)

print("Avg AUC:", np.mean(scoresAUC))
