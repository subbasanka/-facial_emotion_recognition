import gradio as gr

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from keras.layers import Dense 
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K


"""## Reding in the Data Set """

train = pd.read_csv('dataset/fer2013.csv')

train.columns

train['emotion'].value_counts()

train['Usage'].value_counts()

training = train.loc[train['Usage'] == "Training"]
public_test = train.loc[train['Usage'] == "PublicTest"]
private_test = train.loc[train['Usage'] == "PrivateTest"]

print("training set = ", training.shape)
print("General test set = ", public_test.shape)
print("Special test set = ", private_test.shape)

#looking at the Training DataSet
training.head()

data = train.values
y = data[:, 0]
pixels = data[:, 1]
X = np.zeros((pixels.shape[0], 48*48))

for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])

x = X
x = x / 255

"""## Preparing the DataSet

"""

actual_images = []
for i in range(len(training['pixels'])):
    actual_images.append(np.array((training['pixels'][i].split()), dtype=np.float64))

actual_images = np.array(actual_images)

train_labels = training["emotion"]
train_labels = pd.get_dummies(train_labels)

X_train = x[0:28710, :]
Y_train = y[0:28710]
print (X_train.shape, Y_train.shape)
X_crossval = x[28710:32300, :]
Y_crossval = y[28710:32300]

X_train = X_train.reshape((X_train.shape[0], 1 , 48, 48 ))
X_crossval = X_crossval.reshape((X_crossval.shape[0], 1 ,48, 48))

y_ = np_utils.to_categorical(y, 7)
print (y_.shape)
Y_train = y_[:28710]
Y_crossval = y_[28710:32300]

train_pixels = training["pixels"].str.split(" ").tolist()
train_pixels = np.uint8(train_pixels)
train_pixels = train_pixels.reshape((28709, 48, 48, 1))
train_pixels = train_pixels.astype("float32") / 255


private_labels = private_test["emotion"]
private_labels = pd.get_dummies(private_labels)

private_pixels = private_test["pixels"].str.split(" ").tolist()
private_pixels = np.uint8(private_pixels)
private_pixels = private_pixels.reshape((3589, 48, 48, 1))
private_pixels = private_pixels.astype("float32") / 255


public_labels = public_test["emotion"]
public_labels = pd.get_dummies(public_labels)

public_pixels = public_test["pixels"].str.split(" ").tolist()
public_pixels = np.uint8(public_pixels)
public_pixels = public_pixels.reshape((3589, 48, 48, 1))
public_pixels = public_pixels.astype("float32") / 255

from keras.callbacks import ReduceLROnPlateau
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)

target = pd.get_dummies(training['emotion'])
target

train_pixels[0].shape

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.0,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  

datagen.fit(train_pixels)

#looking at one of the image of the training dataset
plt.imshow(np.array((train['pixels'][3].split()), dtype=np.float).reshape(48,48))

"""## Specifying Model Architecture

"""

EarlyStoppingMonitor = EarlyStopping(patience=2)

"""**Model 1** """

#specifying type
model1 = Sequential()
#Sepecifying width and depth
model1.add(Dense(50, activation='relu', input_shape=(2304,)))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(50, activation='relu'))
model1.add(Dense(7, activation='softmax'))
          
#compiling the model
model1.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

"""**Model 2**"""

model2 = Sequential()
model2.add(Dense(100, activation='relu', input_shape=(2304,)))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(7, activation='softmax'))

model2.compile(optimizer = 'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""**Model_3**"""

model_3 = Sequential()

# Conv (evrişim katmanı)
model_3.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
#Ortaklama katmanı
model_3.add(layers.MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

model_3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_3.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model_3.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_3.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_3.add(layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model_3.add(layers.Flatten())

# Tam bağlantı katmanı
model_3.add(layers.Dense(1024, activation='relu'))
model_3.add(layers.Dropout(0.2))
model_3.add(layers.Dense(1024, activation='relu'))
model_3.add(layers.Dropout(0.2))

model_3.add(layers.Dense(7, activation='softmax'))


model_3.compile(optimizer = 'adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

actual_images

"""**Fitting The Model**"""

model1.summary()

"""Model 4"""

private_pixels_m12 = private_pixels.reshape(3589,-1)

#fitting the model 
m1 = model1.fit(actual_images, target,  validation_data = (private_pixels_m12, private_labels), epochs=30, callbacks=[EarlyStoppingMonitor])
#fitting the 2nd model 
m2 = model2.fit(actual_images, target,  validation_data = (private_pixels_m12, private_labels), epochs=30, callbacks=[EarlyStoppingMonitor])

#fitting the 3rd model 
m3 = model_3.fit(train_pixels, train_labels, batch_size = 256, validation_data = (private_pixels, private_labels), epochs=30, callbacks=[EarlyStoppingMonitor])

model_3.save('/content/sample_data/model.h5',save_format='h5')

model_json = model_3.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Create the plot
plt.plot(m1.history['val_loss'], 'r', m2.history['val_loss'], 'b', m3.history['val_loss'], 'g')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def predict_image(img):
  img_4d=img.reshape(-1,48,48,1)
  prediction=model_3.predict(img_4d)[0]
  return {class_names[i]: float(prediction[i]) for i in range(5)}