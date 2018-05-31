import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



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

    #Un-comment the algorithm of your choice
    #The default model is  Naive Bayes Classifier

   # model = GaussianNB().fit(X_train, Y_train) #Naive Bayes
    #model = linear_model.LogisticRegression(C=1e3,random_state=123).fit(X_train, Y_train)#Logisitc Regression Classifier
    #model = svm.SVC(C=1e3,random_state=123).fit(X_train, Y_train)#SVM
    #model = RandomForestClassifier(n_estimators=10,max_leaf_nodes=128,max_depth=32,random_state=123).fit(X_train, Y_train)
    model = MLPClassifier(activation='relu', hidden_layer_sizes=(50), random_state=1, shuffle=True).fit(X_train,Y_train)

    AUC = roc_auc_score(Y_test, model.predict(X_test))
    scoresAUC.append(AUC)
    print(AUC)

print("Avg AUC:", np.mean(scoresAUC))
