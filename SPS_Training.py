import tensorflow as tf
import numpy as np
import os
import cv2
import fnmatch

# for the architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D

# optimizer, data generator and learning rate reductor
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns

# Define Classes
op_classes = {'Stone':0,'Paper':1,'Scissor':2,'None':3}
len_classes = len(op_classes)


# Data preprocessing
dataset_path = 'image_dataset'
X_temp = []
Y_temp = []
for dir in os.listdir(str(dataset_path)):
  label_path = os.path.join(str(dataset_path),dir)
  if not os.path.isdir(label_path):
    continue
  print('Number of images in {} directory are {}'.format(str(dir),len(fnmatch.filter(os.listdir(label_path), '*.jpg'))))
  for imgs in os.listdir(label_path):
    ximg = cv2.imread(os.path.join(label_path,imgs))
    ximg = cv2.cvtColor(ximg,cv2.COLOR_BGR2RGB)
    ximg = cv2.resize(ximg,(227,227))
    X_temp.append(ximg)
    Y_temp.append(dir)
  print('Completed Data preprocessing for {}'.format(str(dir)))

Y = list(map(lambda x: op_classes[x],Y_temp))
X_temp = np.array(X_temp)
Y = np.array(Y)
print('Input Data Shape: ', X_temp.shape)
print('Output Data Shape: ', Y.shape)

# Plot Data Distribution
##value, counts = np.unique(Y_temp, return_counts = True)
##plt.figure(figsize = (6,4))
##sns.barplot(value,counts)
##plt.title('Data Distribution')
##plt.ylabel('Number',fontsize = 12)
##plt.xlabel('Action',fontsize = 12)
##plt.show()

# Display one sample datapoint
#index = 1523
#plt.imshow(X_temp[index])
#plt.title(str(Y_temp[index]))


# Train test Split
X_train, X_test, Y_train, Y_test = train_test_split(X_temp,Y,test_size = 0.25)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
X_train = X_train.astype('float')
X_test = X_test.astype('float')

# One-Hot encoding 
Y_train = np_utils.to_categorical(Y_train,num_classes = len_classes)
Y_test = np_utils.to_categorical(Y_test,num_classes = len_classes)
print('\nOne Hot Encoding performed.')
print('Training Data Input shape: ',X_train.shape)
print('Training Data Output shape: ',Y_train.shape)
print('Testing Data Input shape: ',X_test.shape)
print('Testing Data Output shapes: ',Y_test.shape)

# Define Model
model = Sequential()
dim = 227
model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu', input_shape=(dim,dim,3)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(len_classes, activation='softmax'))

print(model.summary())
# Save Model summary in a text file
with open('modelsummary_Keras_model_test7.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Compile and Fit the Model
print('\nCompile and Fit Model')
model.compile(optimizer= Adam(learning_rate = 0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
batch_size = 16
epochs = 5
# Use 10% of training set as validation set
validation_split = 0.1
history = model.fit(X_train,Y_train,batch_size = batch_size,epochs = epochs,
                    verbose = 2, validation_split = validation_split)

# Plot Loss and Accuracy 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
ax[0].grid(color='black', linestyle='-', linewidth=0.25)
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax[1].grid(color='black', linestyle='-', linewidth=0.25)
legend = ax[1].legend(loc='best', shadow=True)
# plt.show()
plt.savefig('Accuracy_loss_Keras_model_test7.png')

#Evaluating and Saving Model
baseline_model_loss ,baseline_model_accuracy=  model.evaluate(X_test,Y_test,verbose=0)
print('Baseline Test accuracy:',baseline_model_accuracy) 
print('Baseline Test loss:',baseline_model_loss) 

model.save('Keras_model_test7.h5')
print('Saved Model to Disk')
