
# coding: utf-8

# # AMERICAN SIGN LANGUAGE Recognition

# # The Objective of this project is an attempt to a model to identify Static American sign language in real run time 

# ## Problem encountered during model:
# ### Initially when we trained the Convolutional model on directly unprocessed real images model was stuck around accuracy of 40 %

# # Steps to improve our model we took :
# 
#  ### *we took a real world dataset with background subtration in it*
#  ### *this improved our model's accuracy from around 40% to over 90%*
#  ### *To further improve data augmentation which is being done before feeding images to train our model*

# Importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
import h5py
import numpy as np
from keras.preprocessing import image


# In[3]:


# Model Architecture
my_model = Sequential()
my_model.add(Convolution2D(64, (3,  3), input_shape = (64, 64, 3), activation = 'relu'))
my_model.add(MaxPooling2D(pool_size =(2,2)))
my_model.add(Convolution2D(32, (3,  3), activation = 'relu'))
my_model.add(MaxPooling2D(pool_size =(2,2)))
my_model.add(Convolution2D(32, (3,  3), activation = 'relu'))
my_model.add(MaxPooling2D(pool_size =(2,2)))
my_model.add(Flatten())
my_model.add(Dense(128, activation = 'relu'))
my_model.add(Dense(26, activation = 'softmax'))
my_model.compile(
              optimizer = optimizers.SGD(lr = 0.01),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


# Data Augmentation and importing images from directory
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training')

test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
validation_generator = train_datagen.flow_from_directory(
    'mydata/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Fitting model
model = my_model.fit_generator(
        training_set,
        steps_per_epoch=1422,
        epochs=5,
        validation_data = validation_generator,
        validation_steps = 285
      )

## use following code if current model needs to be updated
# import h5py
# my_model.save('/my_model.h5')

# Printing plots 
import matplotlib.pyplot as plt
from keras.models import load_model
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.legend(['train', 'validation'])
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.show()
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.legend(['train', 'validation'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[18]:



# testing on an image
import glob
cnt=0
target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
czn=0
cz=0
net=0
true=[0 for i in range(26)]
false=[0 for i in range(26)]

for i in target_names:
  for name in glob.glob('../asl_recognition_project/mydata/test_set/'+str(i)+'/*'):
  #     print(name)
  #     cnt+=1
    image_path=str(name)
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = my_model.predict(test_image)
    if result[0][cnt]==1:
      true[cnt]+=1
    else:
      false[cnt]+=1
    net+=1
  cnt+=1
accuracy=sum(true)/net
print("Accuracy=",accuracy)


