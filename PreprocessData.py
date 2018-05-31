import numpy as np
from scipy import misc
import glob


print("In progress...")
TCClass = 0.0
TSClass = 1.0
imgDimension = 100

f_handle = open('D:\\ASDDataset_Augmented.csv', 'a')

headerRow = ""
for i in range (0,(imgDimension*imgDimension)):
    headerRow = headerRow + "Pixel" + str(i)+ ","
headerRow = headerRow+"Label"+"\n"

f_handle.write(headerRow)
print("Header Done.")


for file in glob.iglob('D:\\Dataset\\TC\\**\\*.png', recursive=True):

    print("Current File:", file)
    img = misc.imread(file)
    img = img[: ,:, :3]#Excluding the Alpha channel
    img = misc.imresize(img, size=(imgDimension, imgDimension))

    img = (img / 255.0)
    img = img.reshape(-1, 3)

    red = img[:,0]
    green = img[:,1]
    blue = img[:,2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gray = np.append(gray, TCClass)
    gray = gray.reshape(1, (imgDimension*imgDimension)+1)

    #np.savetxt(f_handle, gray, fmt='%s',delimiter=",")
    np.savetxt(f_handle, gray, fmt='%.4g', delimiter=",")

print("TC Done.")


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
print("All Done.")