#!/usr/bin/env python
# coding: utf-8

# In[3]:


from builtins import range, input

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical


# In[26]:


#define size to which images are to be resized
IMAGE_SIZE = [224, 224]

# training config:
epochs = 2
batch_size = 32

#define paths
cardboard_path = 'C:/Users/ASUS/Desktop/pfa/dataset-resized/dataset-resized/cardboard'
glass_path = 'C:/Users/ASUS/Desktop/pfa/dataset-resized/dataset-resized/glass'
metal_path = 'C:/Users/ASUS/Desktop/pfa/dataset-resized/dataset-resized/metal'
paper_path = 'C:/Users/ASUS/Desktop/pfa/dataset-resized/dataset-resized/paper'
plastic_path = 'C:/Users/ASUS/Desktop/pfa/dataset-resized/dataset-resized/plastic'
trash_path = 'C:/Users/ASUS/Desktop/pfa/dataset-resized/dataset-resized/trash'

# Use glob to grab images from path .jpg or jpeg
cardboard_files = glob(cardboard_path + '/*')
glass_files = glob(glass_path + '/*')
metal_files = glob(metal_path  + '/*')
paper_files = glob(paper_path + '/*')
plastic_files = glob(plastic_path + '/*')
trash_files = glob(trash_path + '/*')


# In[5]:


# Visualize file variable contents
print("First 5 cardboard Files: ")
print("Total Count: ",len(cardboard_files))
print("First 5 glass Files: ",glass_files[0:5])
print("Total Count: ",len(glass_files))
print("First 5 metal Files: ",metal_files[0:5])
print("Total Count: ",len(metal_files))
print("First 5 paper Files: ",paper_files[0:5])
print("Total Count: ",len(paper_files))
print("First 5 plastic Files: ",plastic_files[0:5])
print("Total Count: ",len(plastic_files))
print("First 5 trash Files: ",trash_files[0:5])
print("Total Count: ",len(trash_files))


# In[6]:


# Fetch Images and Class Labels from Files
cardboard_labels = []
glass_labels = []
metal_labels = []
paper_labels = []
plastic_labels = []
trash_labels = []

cardboard_images=[]
glass_images=[]
metal_images=[]
paper_images=[]
plastic_images=[]
trash_images=[]

for i in range(len(cardboard_files)):
  image = cv2.imread(cardboard_files[i]) # read file 
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
  image = cv2.resize(image,(224,224)) # resize as per model
  cardboard_images.append(image) # append image
  cardboard_labels.append('cardboard') #append class label
for i in range(len(glass_files)):
  image = cv2.imread(glass_files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  glass_images.append(image)
  glass_labels.append('glass')
for i in range(len(metal_files)):
  image = cv2.imread(metal_files[i]) # read file 
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
  image = cv2.resize(image,(224,224)) # resize as per model
  metal_images.append(image) # append image
  metal_labels.append('metal') #append class label
for i in range(len(paper_files)):
  image = cv2.imread(paper_files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  paper_images.append(image)
  paper_labels.append('paper')
for i in range(len(plastic_files)):
  image = cv2.imread(plastic_files[i]) # read file 
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
  image = cv2.resize(image,(224,224)) # resize as per model
  plastic_images.append(image) # append image
  plastic_labels.append('plastic') #append class label
for i in range(len(trash_files)):
  image = cv2.imread(trash_files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  trash_images.append(image)
  trash_labels.append('trash')


# In[7]:


# look at a random image for fun
def plot_images(images, title):
    nrows, ncols = 5, 8
    figsize = [10, 6]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=(1, 1, 1))

    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i])
        axi.set_axis_off()

    plt.suptitle(title, fontsize=24)
    plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
    plt.show()
plot_images(cardboard_images, 'cardboard')
plot_images(glass_images, 'glass')
plot_images(metal_images, 'metal')
plot_images(paper_images, 'paper')
plot_images(plastic_images, 'plastic')
plot_images(trash_images, 'trash')


# In[8]:


# Convert to array and Normalize to interval of [0,1]
cardboard_images = np.array(cardboard_images) / 255
glass_images = np.array(glass_images) / 255
# Convert to array and Normalize to interval of [0,1]
metal_images = np.array(metal_images) / 255
paper_images = np.array(paper_images) / 255
# Convert to array and Normalize to interval of [0,1]
plastic_images = np.array(plastic_images) / 255
trash_images = np.array(trash_images) / 255


# In[9]:


# Split into training and testing sets for both types of images
cardboard_x_train, cardboard_x_test, cardboard_y_train, cardboard_y_test = train_test_split(
    cardboard_images, cardboard_labels, test_size=0.2)
glass_x_train, glass_x_test, glass_y_train, glass_y_test = train_test_split(
    glass_images, glass_labels, test_size=0.2)
metal_x_train, metal_x_test, metal_y_train, metal_y_test = train_test_split(
    metal_images, metal_labels, test_size=0.2)
paper_x_train, paper_x_test, paper_y_train, paper_y_test = train_test_split(
    paper_images, paper_labels, test_size=0.2)
plastic_x_train, plastic_x_test, plastic_y_train, plastic_y_test = train_test_split(
    plastic_images, plastic_labels, test_size=0.2)
trash_x_train, trash_x_test, trash_y_train, trash_y_test = train_test_split(
    trash_images, trash_labels, test_size=0.2)

# Merge sets for both types of images
X_train = np.concatenate((cardboard_x_train, glass_x_train, metal_x_train, paper_x_train, plastic_x_train, trash_x_train), axis=0)
X_test = np.concatenate((cardboard_x_test, glass_x_test, metal_x_test, paper_x_test, plastic_x_test, trash_x_test), axis=0)
y_train = np.concatenate((cardboard_y_train, glass_y_train, metal_y_train, paper_y_train, plastic_y_train, trash_y_train), axis=0)
y_test = np.concatenate((cardboard_y_test, glass_y_test, metal_y_test, paper_y_test, plastic_y_test, trash_y_test), axis=0)

# Make labels into categories - either 0 .. 5, for our model
y_train = preprocessing.LabelEncoder().fit_transform(y_train)
y_train = to_categorical(y_train)

y_test = preprocessing.LabelEncoder().fit_transform(y_test)
y_test = to_categorical(y_test)


# In[10]:


plot_images(cardboard_x_train, 'X_train cardboard')
plot_images(cardboard_x_test, 'X_test cardboard')
# y_train and y_test contain class lables 0 and 1 representing cardboard and NonCOVID for X_train and X_test
plot_images(glass_x_train, 'X_train glass')
plot_images(glass_x_test, 'X_test glass')
plot_images(metal_x_train, 'X_train metal')
plot_images(metal_x_test, 'X_test metal')
plot_images(paper_x_train, 'X_train paper')
plot_images(paper_x_test, 'X_test paper')
plot_images(plastic_x_train, 'X_train plastic')
plot_images(plastic_x_test, 'X_test plastic')
plot_images(trash_x_train, 'X_train trash')


# In[11]:


# Building Model
vggModel = VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs = vggModel.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(6, activation="sigmoid")(outputs)

model = Model(inputs=vggModel.input, outputs=outputs)

for layer in vggModel.layers:
    layer.trainable = False

model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)

train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# In[12]:


# Visualize Model
model.summary()


# In[13]:


train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# In[27]:


history = model.fit(train_aug.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / batch_size,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs)


# In[28]:


# Save Model and Weights
model.save('vgg_ct.h5')
model.save_weights('vgg_weights_ct.hdf5')


# In[29]:


# Load saved model
model = load_model('vgg_ct.h5')


# In[30]:


y_pred = model.predict(X_test, batch_size=batch_size)


# In[31]:


y_pred


# In[32]:


prediction=y_pred[:]
for index, probability in enumerate(prediction):
  if probability[0] > 0.5:
        plt.title('%.2f' % (probability[0]*100) + '% cardboard')
  elif probability[1] > 0.5:
        plt.title('%.2f' % ((probability[1])*100) + '% glass')
  elif probability[2] > 0.5:
        plt.title('%.2f' % ((probability[2])*100) + '% metal')
  elif probability[3] > 0.5:
        plt.title('%.2f' % ((probability[3])*100) + '% paper')
  elif probability[4] > 0.5:
        plt.title('%.2f' % ((probability[4])*100) + '% plastic')
  elif probability[5] > 0.5:
        plt.title('%.2f' % ((probability[5])*100) + '% trash')
  plt.imshow(X_test[index])
  plt.show()


# In[33]:


# Convert to Binary classes
y_pred_bin = np.argmax(y_pred, axis=1)
y_test_bin = np.argmax(y_test, axis=1)


# In[34]:


def plot_confusion_matrix(normalize):
  classes = ['cardboard','glass','metal','paper','plastic','trash']
  tick_marks = [0.5,1.5,2.5,3.5,4.5,5.5]
  cn = confusion_matrix(y_test_bin, y_pred_bin,normalize=normalize)
  sns.heatmap(cn,cmap='Reds',annot=True)
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

print('Confusion Matrix without Normalization')
plot_confusion_matrix(normalize=None)

print('Confusion Matrix with Normalized Values')
plot_confusion_matrix(normalize='true')


# In[35]:


from sklearn.metrics import classification_report
print(classification_report(y_test_bin, y_pred_bin))


# In[36]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('vgg_ct_accuracy.png')
plt.show()


# In[37]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('vgg_ct_loss.png')
plt.show()


# In[71]:


IMG_SHAPE = 224
import logging
VERBOSE  = 0
def resizeImageForModel(image):
    imageNew = cv2.resize(frame, (IMG_SHAPE,IMG_SHAPE))
    return imageNew

def preProcessImage(image):
    logging.info("preProcessImage: original size" + str(image.shape))
    newimg = cv2.resize(image,(IMG_SHAPE, IMG_SHAPE))
    logging.info("preProcessImage: newsize" + str(newimg.shape))
    newimg = np.array(newimg).astype('float32')/255
    logging.info("preProcess: rescaling values/255")
    if(VERBOSE==1):
        print(newimg)
    return newimg


# In[103]:


from time import gmtime, strftime
import os
#def storLivePredictions(label, image, directory):
 #   print("label " + label)
  #  print("directory " + directory)
   # timestr = time.strftime("%Y%m%d-%H%M%S")
    #filename = "img_" + timestr + ".jpg"
  #  image_filename = os.path.join(directory, filename)
   # print("writing image to " + image_filename)
    #cv2.imwrite(image_filename,image)


# In[100]:


import time
BASE_DIRECTORY ='C:\\Users\\ASUS\\Desktop\\pfa\\'
resultat_directory = os.path.join(BASE_DIRECTORY,"results")
timestr = time.strftime("%Y%m%d-%H%M%S")
resultat_filename ='live_resultat_'+ timestr +'.txt'
resultat_filename = os.path.join(resultat_directory,resultat_filename)
print(resultat_filename)
resultat_file_handle = open(resultat_filename,"w+")
resultat_file_handle.write("NEW RESULTAT: "+ timestr + "\n\n")
print("resultat directory (where will store images) " + resultat_directory)
print("resultat filename (text info) " + resultat_filename)


# In[69]:


def predict(input, model):
    tensor_input = np.expand_dims(input,axis=0)
    print("tensor shape is " + str(tensor_input.shape))
    if(VERBOSE==1):
        print(tensor_input)
    print('\,n# Generate prediction ')
    prediction = model.predict(tensor_input)
    print(prediction)
    return prediction


# In[57]:


def makeDecision(predictions):
    if predictions[0] > 0.5:
        label = "cardboard"
    elif predictions[1] > 0.5:
        label = "glass"
    elif predictions[2] > 0.5:
        label = "metal"
    elif predictions[3] > 0.5:
        label = "paper"
    elif predictions[4] > 0.5:
        label = "plastic"
    elif predictions[5] > 0.5:
        label = "trash"
    else:
        label = "unknown"
    return label
    


# In[115]:


#from IPython import display
#vc = cv2.VideoCapture(0)
#if vc.isOpened():
 #   is_capturing, frame = vc.read()
  #  frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   # webcam_preview = plt.imshow(frame)

#else:
 #   is_capturing = False
#while is_capturing:
 #   try:
  #      is_capturing, frameorig = vc.read()
   #     frale = cv2.cvtColor(frameorig, cv2.COLOR_BGR2RGB)
    #    webcam_preview = plt.imshow(frame)
     #   plt.draw()
      #  image = resizeImageForModel(frame)
       # resize_preview = plt.imshow(image)
        #display.clear_output(wait=True)
        #plt.pause(0.001)
  #      newimageScaled = preProcessImage(frame)
   #     predictions = predict(newimageScaled, model)
    #    print(predictions)
     #   print(predictions[0])
      #  logging.debug(predictions)
       # label = makeDecision(predictions[0])
        #storLivePredictions(label,frameorig,resultat_directory)

    #except KeyboardInterrupt:
     #   vc.release()


# In[ ]:
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    # Continuously capture frames and perform object detection on them
    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = frame1.array
        frame.setflags(write=1)

        # Pass frame into pet detection function
        image = resizeImageForModel(frame)
        resize_preview = plt.imshow(image)
        display.clear_output(wait=True)
        plt.pause(0.001)
        newimageScaled = preProcessImage(frame)
        predictions = predict(newimageScaled, model)
        print(predictions)
        print(predictions[0])
        logging.debug(predictions)
        label = makeDecision(predictions[0])
        cv2.putText(image,"{}".format(label),(30,50),font,1,(255,255,0),2)
        cv2.imshow('tri dechets', image)
        
# Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.release()
cv2.destroyAllWindows()
  


