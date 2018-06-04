#The preprocessing performed on the raw image dataset
# The preprocessing included : Resizing, converting to grayscale, and PCA transformation
#Note: The PCA transformation was exclude from the  CNN model
import numpy as np
from scipy import misc
import glob


TCClass = 0.0
TSClass = 1.0
imgDimension = 100

f_handle = open('D:\\ASDDataset_Augmented.csv', 'a')#The CSV file where data records are appended to

#Formatting the header row of the CSV file
headerRow = ""
for i in range (0,(imgDimension*imgDimension)):
    headerRow = headerRow + "Pixel" + str(i)+ ","
headerRow = headerRow+"Label"+"\n"
f_handle.write(headerRow)


#Processing the TC images
print("Processing TC images...")
for file in glob.iglob('D:\\Dataset\\TC\\**\\*.png', recursive=True):
    print("Current File:", file)
    img = misc.imread(file)
    img = img[: ,:, :3]#Excluding the Alpha channel
    img = misc.imresize(img, size=(imgDimension, imgDimension))#Resizing images

    img = (img / 255.0)
    img = img.reshape(-1, 3)

    red = img[:,0]
    green = img[:,1]
    blue = img[:,2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue #Converting into grayscale
    gray = np.append(gray, TCClass)
    gray = gray.reshape(1, (imgDimension*imgDimension)+1)
    np.savetxt(f_handle, gray, fmt='%.4g', delimiter=",")

print("TC Done.")

#Same process for TS images
for file in glob.iglob('D:\\Dataset\\TS\\**\\*.png', recursive=True):
    print("Current File:", file)
    img = misc.imread(file)
    img = img[:,:,:3]#Excluding the Alpha channel
    img = misc.imresize(img, size=(imgDimension, imgDimension))

    img = (img / 255.0)
    img = img.reshape(-1, 3)
    red = img[:,0]
    green = img[:,1]
    blue = img[:,2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gray = np.append(gray, TSClass)
    gray = gray.reshape(1, (imgDimension*imgDimension)+1)

    np.savetxt(f_handle, gray, fmt='%.4g',delimiter=",")
f_handle.close()

#PCA transformation
my_data = np.genfromtxt('D:\\ASDDataset_Augmented.csv', delimiter=',')
X = my_data[1:,0:10000]
labels = my_data[1:,10000].astype(dtype='int')

nDim = 50
pca = PCA(n_components=nDim)
X_transformed = pca.fit(X).transform(X)