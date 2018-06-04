#A simple DNN model of three hidden layer is developed in this code
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np


dataset = np.genfromtxt('D:\\ASDDataset_Processed.csv', delimiter=',')

#print(dataset.shape)
features = dataset[1:,:40000]
labels = dataset[1:,40000].astype(dtype='int')
#print(labels)

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30, random_state=42)


model = Sequential()
model.add(Dense(500, input_dim=(40000), activation='relu'))

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fit the model
model.fit(X_train, Y_train, batch_size=12, nb_epoch=100, verbose=1)

#Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
AUC = roc_auc_score(Y_test, model.predict(X_test))
print(AUC) #0.8119512885868027 on non-augmented dataset
