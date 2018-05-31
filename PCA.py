import numpy as np
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(7)
my_data = np.genfromtxt('D:\\ASDDataset_Augmented.csv', delimiter=',')


X = my_data[1:,0:10000]
labels = my_data[1:,10000].astype(dtype='int')

nDim = 50
pca = PCA(n_components=nDim)
X_transformed = pca.fit(X).transform(X)


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
scoresAUC = []
for train, test in kfold.split(X_transformed, labels):
    model = Sequential()
    model.add(Dense(80, input_dim=(nDim), activation='relu', name='L0'))

    model.add(Dense(40, activation='relu', name='L1'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X_transformed[train], labels[train], batch_size=12, nb_epoch=100, verbose=0)

    # 10. Evaluate model on test data
    AUC = roc_auc_score(labels[test], model.predict(X_transformed[test]))
    print(AUC)
    scoresAUC.append(AUC)


print("Avg AUC:", np.mean(scoresAUC))