import cv2
import glob, os

negDir = 'D:/images_cropped/neg/'
posDir = 'D:/images_cropped/pos/'
imgDim = 224
for imgFile in glob.iglob('D:/images/**/*.jpg', recursive=True):
    #print(imgFile)
    imgID = os.path.basename(imgFile).split(".jpg")[0]

    img = cv2.imread(imgFile)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Thresholding
    th, threshed = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY)
    #Find the max-area contour
    _, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    imgCropped = img[y:y + h, x:x + w]#Cropped image

    imgCropped = cv2.resize(imgCropped, (imgDim, imgDim))#resize
    #Saving cropped image
    if 'neg' in imgFile:
        cv2.imwrite(negDir + imgID + '.jpg', imgCropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif 'pos'in imgFile:
        cv2.imwrite(posDir + imgID + '.jpg', imgCropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
