
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters
import pytesseract
from PIL import Image

# example of loading an image with the Keras API
from keras.preprocessing import image
import matplotlib.pyplot as plt
df_train = pd.read_csv("Train_metadata.csv")
df_test = pd.read_csv("Test_metadata.csv")

#for i in range(9):
#	# define subplot
#    plt.subplot(330 + 1 + i);
#    img = image.load_img(df_train.iloc[i*59][0]);
#    x = image.img_to_array(img)
#    plt.imshow(x)
# show the figure
plt.show()
width_tr = []
height_tr = []
channel_tr = []
width_te = []
height_te = []
channel_te = []
formatt = []
for row in df_train.iterrows():
    path = row[1]['File_Path']
    label = row[1]['Class']
    try:
        img = Image.open(path)
        formatt.append(img.format)
    except Exception as e:
        print(e)
for row in df_train.iterrows():
    path = row[1]['File_Path']
    label = row[1]['Class']
    try:
      
        img = image.load_img(path)
        
        # convert to 3D tensor with shape (224, 224, 3) with 3 RGB channels
        x = image.img_to_array(img)
        width_tr.append(x.shape[0])
        height_tr.append(x.shape[1])
        channel_tr.append(x.shape[2])
    except:
        pass
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return it
for row in df_test.iterrows():
    path = row[1]['File_Path']
    try:
        
        img = image.load_img(path)
        
        # convert to 3D tensor with shape (224, 224, 3) with 3 RGB channels
        x = image.img_to_array(img)
        width_te.append(x.shape[0])
        height_te.append(x.shape[1])
        channel_te.append(x.shape[2])
    
    except:
        pass
plt.scatter(width_te, height_te, color = 'red', label = 'Test Data')
plt.xlabel('Width')
plt.ylabel('Height')
plt.legend()
plt.show()
plt.scatter(width_tr, height_tr, color = 'blue', label = 'Train Data')
plt.xlabel('Width')
plt.ylabel('Height')
plt.legend()

plt.show()

imag = cv2.imread(df_train.iloc[101,0], 0)

#plt.figure(figsize=(8, 8))
#plt.imshow(imag)
#sobel_image = filters.sobel(imag)
#plt.figure(figsize=(8, 8))
#plt.imshow(sobel_image)
#plt.axis("off")
#plt.show()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
label_string = []
for row in df_test.iterrows():
    img = cv2.imread(row[1]['File_Path'])
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#    threshold_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)[1]

    threshold_img = cv2.threshold(gray_image,160,255,cv2.THRESH_BINARY)[1]

    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
   
    label_string.append(pytesseract.image_to_string(threshold_img, config=(" --psm 6"))+pytesseract.image_to_string(threshold_img, config=(" --psm 7"))+pytesseract.image_to_string(threshold_img, config=(" --psm 8"))+pytesseract.image_to_string(threshold_img, config=(" --psm 9")))
    plt.figure()
    plt.imshow(threshold_img, cmap = 'gray')
    
labels = df_test['Class'].drop_duplicates()
labels = labels.to_list()

for i in labels:
    i = i.lower()
    for j in label_string:
        j = j.lower()
        
        
myDict = {key: [] for key in labels}
for i in labels:

    iff = i.lower()
    for j in label_string:
        how_many = 0
        j = j.lower()
        for k in [iff[i:i+4] for i in range(0, len(iff)-3)]:
            how_many += j.count(k)
            
        if how_many >0:
            myDict[i].append(1)
        else:
            myDict[i].append(0)
label_Dict = {key: [] for key in labels}
     
for key, value in myDict.items():
    if (np.argwhere(np.array(value)>0).shape[0])>0:
        for i in range(0, np.argwhere(np.array(value)>0).shape[0]):
            label_Dict[key].append(np.argwhere(np.array(value)>0)[i][0])
items_correctly_predicted = 0
for key, value in label_Dict.items():
    for i in value:
        df_test.drop([i], inplace= True)
        items_correctly_predicted +=1