from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import misc
import glob
import sklearn.metrics
import matplotlib.pyplot as plt

def PreprocessData():

    print("Precessing Images...")
    TCClass = 0.0
    TSClass = 1.0
    imgDimension = 200
    crop = 100
    dataset = np.empty(shape=[0,(imgDimension*imgDimension)+1], dtype=float)
    for file in glob.iglob('D:\\Dataset\\TC\\**\\*.png', recursive=True):
        print("Current File:", file)
        img = misc.imread(file)
        img = img[:img.shape[0] - crop, :img.shape[1] - crop, :3]#Cropping
        img = misc.imresize(img, size=(imgDimension, imgDimension))
        img = (img / 255.0)
        img = img.reshape(-1, 3)
        red = img[:,0]
        green = img[:,1]
        blue = img[:,2]
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        gray = np.append(gray, TCClass)
        gray = gray.reshape(1, (imgDimension*imgDimension)+1)
        dataset = np.append(dataset, gray, axis=0)

    print("TC Done.")
    for file in glob.iglob('D:\\Dataset\\TS\\**\\*.png', recursive=True):
        print("Current File:", file)
        img = misc.imread(file)
        img = img[:img.shape[0] - crop, :img.shape[1] - crop, :3]  # Cropping
        img = misc.imresize(img, size=(imgDimension, imgDimension))
        img = (img / 255.0)
        img = img.reshape(-1, 3)

        red = img[:, 0]
        green = img[:, 1]
        blue = img[:, 2]
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
        gray = np.append(gray, TSClass)
        gray = gray.reshape(1, (imgDimension*imgDimension)+1)
        dataset = np.append(dataset, gray, axis=0)
    print("All Done.")

    return dataset

dataset = PreprocessData()
#print(dataset.shape)
features = dataset[:, :dataset.shape[1]-1]
labels = np.int_(dataset[:, dataset.shape[1]-1].reshape(-1,1))

#print(features.shape)
#print(labels.shape)
#print(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
DNN = MLPClassifier( activation='relu', hidden_layer_sizes=(100,50), random_state=1,shuffle =True)
DNN.fit(X_train, y_train)
predictions = DNN.predict(X_test)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, predictions)
#print(fpr)
#print(tpr)
roc_auc = sklearn.metrics.auc(fpr,tpr)
print(roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
