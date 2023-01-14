# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 11:29:19 2022

@author: ankit
"""
#https://towardsdatascience.com/implementing-ssd-in-keras-part-i-network-structure-da3323f11cff
import numpy as np
import keras as k 
import pandas as pd
import inspect
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf

# example of loading an image with the Keras API
from keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
model_dictionary = {m[0]:m[1] for m in inspect.getmembers(tf.keras.applications, inspect.isfunction)}

df_train = pd.read_csv("Train_metadata.csv")
df_test = pd.read_csv("Test_metadata.csv")
#train_files, val_files, train_targets, val_targets1 = train_test_split(df_train['File_Path'], df_train['Class'], test_size=0.2)
#train_files = train_files.reset_index(drop = True)
#
#train_targets = train_targets.reset_index(drop = True)
#val_files = val_files.reset_index(drop = True)
#
#val_targets = val_targets1.reset_index(drop = True)
train_files = df_train['File_Path']
train_targets = df_train['Class']
test_files = df_test['File_Path']
test_targets = df_test['Class']
label_encoder = LabelEncoder()
label_encoder.fit(train_targets)
integer_encoded_train = label_encoder.transform(train_targets)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded_train = integer_encoded_train.reshape(len(integer_encoded_train), 1)
onehot_encoder.fit(integer_encoded_train)
train_targets = onehot_encoder.transform(integer_encoded_train)

integer_encoded_test = label_encoder.transform(test_targets)
integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)

test_targets = onehot_encoder.transform(integer_encoded_test)
#
#integer_encoded_val = label_encoder.transform(val_targets)
#integer_encoded_val = integer_encoded_val.reshape(len(integer_encoded_val), 1)
#
#val_targets = onehot_encoder.transform(integer_encoded_val)


#create a checkpointer to save the CNN model with the best weight parameters

checkpointer = ModelCheckpoint(filepath='saved_keras_models/weights.best.CNN.hdf5', 
                               save_best_only=True)

#def path_to_tensor(train_files):

def process(files, n):
    cant = []
    list_of_tensors = []
    for i in  files:
    # loads RGB image as PIL format with 224x224 pixels
        try:
            img = image.load_img(i, target_size=(224, 224))
            # convert to 3D tensor with shape (224, 224, 3) with 3 RGB channels
            x = image.img_to_array(img)
            # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return it
            
            list_of_tensors.append(np.expand_dims(x, axis=0)/255)
        except:
            cant.append(files[files == i].index[0])
    return np.vstack(list_of_tensors), cant
#    return 

train_tensors, cant_train =  process(train_files, 224)
for i in cant_train:
    print(i)
    train_targets = np.delete(train_targets, i,0)
    
#val_tensors, cant_val =  process(val_files, 224)
#for i in cant_val:
#    val_targets = np.delete(val_targets, i,0)
#    
test_tensors, cant_test =  process(test_files, 224)
for i in cant_test:
    test_targets = np.delete(test_targets, i,0)

##################### NEW MODEL############################################################
    
#cnn_model = Sequential()
#
##after the input layer, add the first convolutional layer with 32 2x2-filters 
#cnn_model.add(Conv2D (kernel_size = (2,2), filters = 32, 
#                      input_shape=train_tensors.shape[1:], activation='relu'))
##add a max pooling layer with a 2x2 pooling window
#cnn_model.add(MaxPooling2D(pool_size=2))
##add the second convolutional layer with 64 2x2-filters 
#cnn_model.add(Conv2D(kernel_size = 2, filters = 64, activation='relu'))
#cnn_model.add(MaxPooling2D(pool_size=2))
##add the third convolutional layer with 128 2x2-filters 
#cnn_model.add(Conv2D(kernel_size = 2, filters = 128, activation='relu'))
##add a dropout layer so that each node has a chance of 20% to be dropped when training
#cnn_model.add(Dropout(0.2))
#cnn_model.add(MaxPooling2D(pool_size = 2))
##add a global average pooling layer
#cnn_model.add(GlobalAveragePooling2D())
##add the final fully connected output layer with 109 node for all 109 logo classes
#cnn_model.add(Dense(18, activation = 'softmax'))
#cnn_model.summary()
#
#cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#,
##Train the model for 100 epochs
#H = cnn_model.fit(train_tensors, train_targets, epochs=100, batch_size = 8, verbose=2, validation_data=(val_tensors, val_targets), callbacks=[checkpointer])
#
##After training, load the model with the minimal error on the validation set
#cnn_model.load_weights('saved_keras_models/weights.best.CNN.hdf5')

model_benchmarks = {'model_name': [], 'num_model_params': [], 'validation_accuracy': []}
not_model = ['NASNetLarge', 'Xception', 'ResNet101', 'ResNet101V2','MobileNetV3Small',' EfficientNetB0','MobileNetV3Large','EfficientNetB1', 'EfficientNetB2','EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetB0','VGG16', 'VGG19', 'ResNet50', 'ResNet152', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetMobile', 'ResNet152V2'  ]
for model_name, model in tqdm(model_dictionary.items()):
    # Special handling for "NASNetLarge" since it requires input images with size (331,331)
    if model_name in not_model:
        print("skip")
    else: 
       
            
        # load the pre-trained model with global average pooling as the last layer and freeze the model weights
        pre_trained_model = model(include_top=False, pooling='avg', input_shape=(224, 224, 3))
        pre_trained_model.trainable = False
        
        # custom modifications on top of pre-trained model and fit
        clf_model = tf.keras.models.Sequential()
        clf_model.add(pre_trained_model)
        
        clf_model.add(tf.keras.layers.Dense(18, activation='softmax'))
        clf_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        history = clf_model.fit(train_tensors, train_targets, epochs=100, validation_data=(test_tensors, test_targets))
        
        # Calculate all relevant metrics
        model_benchmarks['model_name'].append(model_name)
        model_benchmarks['num_model_params'].append(pre_trained_model.count_params())
        model_benchmarks['validation_accuracy'].append(history.history['val_accuracy'][-1])

benchmark_df = pd.DataFrame(model_benchmarks)

# sort in ascending order of num_model_params column
benchmark_df.sort_values('num_model_params', inplace=True)
benchmark_df.to_csv("benchmark_df_3.csv")
# write results to csv  index=False)

#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0,100),H.history["accuracy"],label="train_acc")
#plt.plot(np.arange(0,100),H.history["val_accuracy"],label="val_acc")
#plt.title("Training and Validation Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Accuracy")
#plt.legend(loc="upper left")
#plt.show()
#test_accuracy = cnn_model.evaluate(test_tensors, test_targets, verbose=0)[1]
#print("Test Accuracy " +str(test_accuracy))