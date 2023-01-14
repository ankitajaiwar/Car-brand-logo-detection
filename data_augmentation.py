# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:17:55 2022

@author: ankit
"""

# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import pandas as pd
import os
from imgaug import augmenters as iaa

save_here = r".\\Augmented"
df = pd.read_csv("Train_metadata.csv")

seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0.0, 1.0))])
samples = df.shape[0]
for row in df.iterrows():
    path = row[1]['File_Path']
    label = row[1]['Class']
    try:
        img = load_img(path)
    except:
        continue
# convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    image = expand_dims(data, 0)
    # create image data augmentation generator
    
    
    datagen = ImageDataGenerator(
   # divide each input by its std
          # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = [0.8,1.6], # Randomly zoom image 
        preprocessing_function=seq.augment_image
        

        )  # randomly flip images

    # prepare iterator
    datagen.fit(image)
    isExist = os.path.exists(save_here+'\\'+str(label))
    if not isExist:
  
  # Create a new directory because it does not exist 
      os.makedirs(save_here+'\\'+str(label))
    for x, val in zip(datagen.flow(image, save_to_dir=save_here+'\\'+str(label), save_prefix='aug',save_format='png'),range(5)) :     
        pass
    # generate samples and plot
