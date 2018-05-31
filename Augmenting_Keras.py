from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import glob


augmenter = ImageDataGenerator(
    rotation_range=10,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

imgCount=5
print("Augmenting TC Samples:")
for file in glob.iglob('D:\\Dataset\\TC\\Raw\\*.png', recursive=True):
    print("Current File:", file)
    img = load_img(file)  # this is a PIL image
    img = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    img = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0
    for batch in augmenter.flow(img, batch_size=1,
                              save_to_dir='D:\\Dataset\\TC\\Augmented', save_prefix='TC_Aug', save_format='png'):
        i += 1
        if i == imgCount:
            break

#Augmenting TS images
print("Augmenting TS Samples:")
for file in glob.iglob('D:\\Dataset\\TS\\Raw\\*.png', recursive=True):
    print("Current File:", file)
    img = load_img(file)  # this is a PIL image
    img = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    img = img.reshape((1,) + img.shape)  #Numpy array with shape (1, 3, 150, 150)

    i = 0
    for batch in augmenter.flow(img, batch_size=1,
                              save_to_dir='D:\\Dataset\\TS\\Augmented', save_prefix='TS_Aug', save_format='png'):
        i += 1
        if i == imgCount:
            break


