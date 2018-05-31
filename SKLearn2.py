from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
#import sklearn as sk

#print ("sklearn version :", sk.__version__)

dataset = pd.read_csv('D:\\ASDDataset_Processed.csv')
features = dataset.iloc[:, 0:dataset.shape[1]-1].as_matrix()
#print(features.shape)
labels = dataset.loc[:, ['Label']].astype(int).as_matrix()

DNN = MLPClassifier( activation='relu',learning_rate_init=0.1,max_iter =200 ,hidden_layer_sizes=(100,50), random_state=1,shuffle =True)
scores = cross_val_score(DNN, features, labels, cv=3, scoring='roc_auc')
print("Mean AUC (3-Fold):", scores.mean())

#roc_auc = sklearn.metrics.auc(fpr,tpr)
#print(roc_auc)

#plt.figure()
#lw = 2
#plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.legend(loc="lower right")
#plt.show()




